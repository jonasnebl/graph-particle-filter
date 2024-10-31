#include "particleTracker.h"

#include <hungarian.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <vector>

#include "agent.h"
#include "particle.h"
#include "simulation.h"
#include "warehouse_data.h"

ParticleTracker::ParticleTracker(double T_step, int N_tracks_init, int N_particles)
    : T_step(T_step), N_tracks(N_tracks), N_particles(N_particles) {
    // --- init particles ---
    for (int i = 0; i < N_tracks; i++) {
        std::vector<Particle> particles_per_human;
        for (int j = 0; j < N_particles; j++) {
            particles_per_human.push_back(Particle(&graph));
        }
        particles.push_back(particles_per_human);
    }

    // --- init particle weights ---
    particle_weights = std::vector<double>(N_particles, 1.0 / N_particles);

    // --- random number generator ---
    std::random_device rd;
    mt = std::mt19937(rd());

    // --- init N_perceived window so that the tracker start's with N_tracks_init tracks ---
    std::discrete_distribution<int> N_perceived_distribution(
        graph.N_perceived_likelihood_matrix[N_tracks_init].begin(),
        graph.N_perceived_likelihood_matrix[N_tracks_init].end());
    for (int i = 0; i < T_WINDOW / T_step; i++) {
        N_perceived_window.push_back(N_perceived_distribution(mt));
    }
}

std::vector<double> ParticleTracker::add_merged_perceptions(
    std::vector<pybind11::dict> perceived_humans, std::vector<Point> robot_positions) {
    // --- update particles for each human individually ---
    for (int i = 0; i < N_particles; i++) {
        std::vector<std::vector<int>> assignment_cost_matrix =
            calc_assignment_cost_matrix(perceived_humans, i);
        std::vector<int> hypotheses =
            assign_perceived_humans_to_internal_humans(assignment_cost_matrix);
        double likelihood = 1.0;
        for (int j = 0; j < N_tracks; j++) {
            int perception_index = hypotheses[j];
            if (perception_index == -1) {
                likelihood *= particles[j][i].likelihood_no_perception(robot_positions);
            } else {
                particles[j][i] = generate_new_particle_from_perception(
                    perceived_humans[perception_index]["position"].cast<Point>(),
                    perceived_humans[perception_index]["position_stddev"].cast<double>(),
                    perceived_humans[perception_index]["heading"].cast<double>(),
                    perceived_humans[perception_index]["heading_stddev"].cast<double>());
            }
        }
        particle_weights[i] *= likelihood;  // sequential importance sampling
    }
    normalize_weights();

    // --- resample particles ---
    const double resample_threshold = 1e-2 / static_cast<double>(N_particles);
    std::discrete_distribution<int> resample_distribution(particle_weights.begin(),
                                                          particle_weights.end());
    for (int i = 0; i < N_particles; i++) {
        int random_source_particle_index = resample_distribution(mt);
        // --- copy particles ---
        if (particle_weights[i] < resample_threshold) {
            for (int j = 0; j < N_tracks; j++) {
                // use the copy constructor to reseed the random number generator
                particles[j][i] = Particle(particles[j][random_source_particle_index]);
            }
        }
        // --- copy weight ---
        particle_weights[i] = particle_weights[random_source_particle_index];
    }
    normalize_weights();

    // // --- print effective sample size for debugging purposes ---
    // std::cout << "Effective sample size: " << calc_effective_sample_size()
    //           << "; N_particles: " << N_particles << std::endl;

    // --- calculate edge probabiliities from internal state ---
    return calc_edge_probabilities();
}

int ParticleTracker::estimate_N_humans(int N_perceived) {
    N_perceived_window.push_back(N_perceived);
    N_perceived_window.pop_front();
    std::vector<double> N_estimated_likelihood(graph.N_perceived_likelihood_matrix.size(), 1.0);
    for (int i = 0; i < N_perceived_window.size(); i++) {
        for (int j = 0; j < graph.N_perceived_likelihood_matrix.size(); j++) {
            N_estimated_likelihood[j] *=
                graph.N_perceived_likelihood_matrix[j][N_perceived_window[i]];
        }
        // normalize to avoid float point underflow
        double sum =
            std::accumulate(N_estimated_likelihood.begin(), N_estimated_likelihood.end(), 0.0);
        for (int j = 0; j < graph.N_perceived_likelihood_matrix.size(); j++) {
            N_estimated_likelihood[j] /= sum;
        }
    }
    return std::max_element(N_estimated_likelihood.begin(), N_estimated_likelihood.end()) -
           N_estimated_likelihood.begin();
}

void ParticleTracker::add_one_track() {
    N_tracks++;
    std::vector<Particle> particles_new_track;
    for (int i = 0; i < N_particles; i++) {
        particles_new_track.push_back(Particle(&graph));
    }
    particles.push_back(particles_new_track);
}

void ParticleTracker::remove_one_track() {
    N_tracks--;
    particles.pop_back();  // remove last track
}

std::pair<std::vector<pybind11::dict>, std::vector<Point>> ParticleTracker::merge_perceptions(
    std::vector<pybind11::dict> robot_perceptions) {
    std::pair<std::vector<pybind11::dict>, std::vector<Point>> merged_perceptions;
    for (const auto& perception : robot_perceptions) {
        Point robot_position = perception["position"].cast<Point>();
        merged_perceptions.second.push_back(robot_position);
        auto perceived_humans = perception["perceived_humans"].cast<std::vector<pybind11::dict>>();
        std::vector<pybind11::dict> new_perceptions_this_robot;
        for (const auto& perceived_human : perceived_humans) {
            Point perceived_pos = perceived_human["position"].cast<Point>();
            double perceived_heading = perceived_human["heading"].cast<double>();
            double perceived_pos_variance =
                std::pow(XY_STDDEV * Agent::euclidean_distance(robot_position, perceived_pos), 2);
            double perceived_heading_variance = std::pow(HEADING_STDDEV, 2);

            bool matching_perception_found = false;
            for (auto& merged_human : merged_perceptions.first) {
                Point merged_pos = merged_human["position"].cast<Point>();
                double merged_heading = merged_human["heading"].cast<double>();
                double merged_pos_variance =
                    std::pow(merged_human["position_stddev"].cast<double>(), 2);
                double merged_heading_variance =
                    std::pow(merged_human["heading_stddev"].cast<double>(), 2);

                double eucl_distance = Agent::euclidean_distance(perceived_pos, merged_pos);
                double head_distance = heading_distance(perceived_heading, merged_heading);
                if (eucl_distance <
                        4 * (std::sqrt(merged_pos_variance) + std::sqrt(perceived_pos_variance)) &&
                    head_distance < 4 * (std::sqrt(merged_heading_variance) +
                                         std::sqrt(perceived_heading_variance))) {
                    // update position and heading
                    Point updated_position;
                    updated_position.first = (merged_pos_variance * perceived_pos.first +
                                              perceived_pos_variance * merged_pos.first) /
                                             (merged_pos_variance + perceived_pos_variance);
                    updated_position.second = (merged_pos_variance * perceived_pos.second +
                                               perceived_pos_variance * merged_pos.second) /
                                              (merged_pos_variance + perceived_pos_variance);
                    merged_human["position"] = updated_position;
                    merged_human["position_stddev"] =
                        std::sqrt(merged_pos_variance * perceived_pos_variance /
                                  (merged_pos_variance + perceived_pos_variance));

                    double updated_heading = (merged_heading_variance * perceived_heading +
                                              perceived_heading_variance * merged_heading) /
                                             (merged_heading_variance + perceived_heading_variance);
                    merged_human["heading"] = updated_heading;
                    merged_human["heading_stddev"] =
                        std::sqrt(merged_heading_variance * perceived_heading_variance /
                                  (merged_heading_variance + perceived_heading_variance));
                    matching_perception_found = true;
                    break;
                }
            }
            if (!matching_perception_found) {
                new_perceptions_this_robot.push_back(perceived_human);
                new_perceptions_this_robot.back()["position_stddev"] =
                    std::sqrt(perceived_pos_variance);
                new_perceptions_this_robot.back()["heading_stddev"] =
                    std::sqrt(perceived_heading_variance);
            }
        }
        merged_perceptions.first.insert(merged_perceptions.first.end(),
                                        new_perceptions_this_robot.begin(),
                                        new_perceptions_this_robot.end());
    }
    return merged_perceptions;
}

double ParticleTracker::heading_distance(double h1, double h2) {
    double heading_difference = std::fmod(h1 - h2 + M_PI, 2 * M_PI);
    if (heading_difference < 0) {
        heading_difference += 2 * M_PI;
    }
    return std::abs(heading_difference - M_PI);
}

Particle ParticleTracker::generate_new_particle_from_perception(Point perceived_pos,
                                                                double position_stddev,
                                                                double perceived_heading,
                                                                double heading_stddev) {
    std::normal_distribution<double> position_noise(0, position_stddev);
    std::normal_distribution<double> heading_noise(0, heading_stddev);

    Point noisy_perceived_pos = perceived_pos;
    noisy_perceived_pos.first += position_noise(mt);
    noisy_perceived_pos.second += position_noise(mt);
    double noisy_perceived_heading = perceived_heading + heading_noise(mt);

    std::tuple<int, double> belonging_edge_and_t =
        get_belonging_edge(noisy_perceived_pos, noisy_perceived_heading);
    int belonging_edge = std::get<0>(belonging_edge_and_t);
    double t = std::get<1>(belonging_edge_and_t);
    return Particle(belonging_edge, t, &graph);
}

std::tuple<int, double> ParticleTracker::get_belonging_edge(Point position, double heading) {
    std::vector<double> distances;
    std::vector<double> t_values;
    for (int i = 0; i < graph.edges.size(); i++) {
        std::tuple<double, double> edge_to_pose_distance_and_t_ =
            edge_to_pose_distance_and_t(i, position, heading, graph);
        distances.push_back(std::get<0>(edge_to_pose_distance_and_t_));
        t_values.push_back(std::get<1>(edge_to_pose_distance_and_t_));
    }
    int min_index = std::min_element(distances.begin(), distances.end()) - distances.begin();
    return std::make_tuple(min_index, t_values[min_index]);
}

std::tuple<double, double> ParticleTracker::edge_to_pose_distance_and_t(int edge, Point position,
                                                                        double heading,
                                                                        graph_struct& graph) {
    Point edge_start = graph.nodes[graph.edges[edge].first];
    Point edge_end = graph.nodes[graph.edges[edge].second];

    // --- calculate cartesian distance from edge to position ---
    const double dx = edge_end.first - edge_start.first;
    const double dy = edge_end.second - edge_start.second;
    const double l2 = dx * dx + dy * dy;  // squared length of the edge
    if (l2 == 0.0) {                      // edge_start == edge_end
        return std::make_tuple(
            std::hypot(position.first - edge_start.first, position.second - edge_start.second),
            0.0);
    }
    const double t = std::max(0.0, std::min(1.0, ((position.first - edge_start.first) * dx +
                                                  (position.second - edge_start.second) * dy) /
                                                     l2));
    const Point projection = {edge_start.first + t * dx, edge_start.second + t * dy};
    double cartesian_distance =
        std::hypot(position.first - projection.first, position.second - projection.second);

    // --- calculate heading distance from edge to heading ---
    double edge_heading = std::atan2(dy, dx);
    double heading_dist = heading_distance(heading, edge_heading);

    // --- return weighted sum of cartesian distance and heading difference ---
    return std::make_tuple(cartesian_distance + HEADING_WEIGHT * heading_dist, t);
}

std::vector<double> ParticleTracker::predict() {
    for (int i = 0; i < N_tracks; i++) {
        for (int j = 0; j < N_particles; j++) {
            particles[i][j].predict(T_step);
        }
    }
    return calc_edge_probabilities();
}

void ParticleTracker::normalize_weights() {
    double sum_weights = std::accumulate(particle_weights.begin(), particle_weights.end(), 0.0);
    for (int j = 0; j < N_particles; j++) {
        particle_weights[j] /= sum_weights;
        if (std::isnan(particle_weights[j])) {
            throw std::runtime_error("NaN detected in particle weights");
        }
    }
}

void ParticleTracker::print_weights() {
    for (int i = 0; i < N_particles; i++) {
        std::cout << particle_weights[i] << " ";
    }
    std::cout << std::endl;
}

std::vector<double> ParticleTracker::calc_edge_probabilities() {
    std::vector<double> edge_probabilities(graph.edges.size(), 0.0);
    for (int i = 0; i < N_particles; i++) {
        for (int j = 0; j < graph.edges.size(); j++) {
            for (int k = 0; k < N_tracks; k++) {
                if (particles[k][i].is_human_on_edge(j)) {
                    edge_probabilities[j] += particle_weights[i];
                    break;
                }
            }
        }
    }
    return edge_probabilities;
}

std::vector<std::vector<int>> ParticleTracker::calc_assignment_cost_matrix(
    std::vector<pybind11::dict> perceived_humans, int particle_index) {
    std::vector<std::vector<int>> cost_matrix(N_tracks, std::vector<int>(N_tracks, 0));
    for (int i = 0; i < N_tracks; i++) {      // tracks
        for (int j = 0; j < N_tracks; j++) {  // perceived humans
            if (j < perceived_humans.size()) {
                auto particle = particles[i][particle_index];
                auto perceived_human = perceived_humans[j];
                double assignment_cost =
                    particle.assignment_cost(perceived_human["position"].cast<Point>(),
                                             perceived_human["heading"].cast<double>());
                cost_matrix[i][j] +=
                    static_cast<int>(1e4 * assignment_cost);  // library expects integers
            } else {
                cost_matrix[i][j] = 1e8;
            }
        }
    }
    return cost_matrix;
}

std::vector<int> ParticleTracker::assign_perceived_humans_to_internal_humans(
    std::vector<std::vector<int>> cost_matrix_input) {
    // --- allocate and fill c style cost matrix for hungarian algorithm lib ---
    int** cost_matrix = (int**)calloc(N_tracks, sizeof(int*));
    for (int i = 0; i < N_tracks; i++) {
        cost_matrix[i] = (int*)calloc(N_tracks, sizeof(int));
        for (int j = 0; j < N_tracks; j++) {
            cost_matrix[i][j] = cost_matrix_input[i][j];
        }
    }

    // --- assign perceived humans to internal humans using the hungarian algorithm ---
    hungarian_problem_t prob;
    int matrix_size =
        hungarian_init(&prob, cost_matrix, N_tracks, N_tracks, HUNGARIAN_MODE_MINIMIZE_COST);
    hungarian_solve(&prob);
    std::vector<int> perceived_human_per_internal_human(N_tracks, -1);
    for (int i = 0; i < N_tracks; i++) {
        for (int j = 0; j < N_tracks; j++) {
            if (prob.assignment[i][j] == 1) {
                if (cost_matrix[i][j] < 0.5e8) {
                    perceived_human_per_internal_human[i] = j;
                } else {
                    perceived_human_per_internal_human[j] = -1;
                }
                break;
            }
        }
    }
    hungarian_free(&prob);
    for (int i = 0; i < N_tracks; i++) {
        free(cost_matrix[i]);
    }
    return perceived_human_per_internal_human;
}

double ParticleTracker::calc_effective_sample_size() const {
    return 1.0 / std::accumulate(particle_weights.begin(), particle_weights.end(), 0.0,
                                 [](double sum, double weight) { return sum + weight * weight; });
}