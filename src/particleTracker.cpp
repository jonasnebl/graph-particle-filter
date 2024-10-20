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

ParticleTracker::ParticleTracker(double T_step, int N_humans_max, int N_particles)
    : T_step(T_step), N_humans_max(N_humans_max), N_particles(N_particles) {
    // init particles
    for (int i = 0; i < N_humans_max; i++) {
        std::vector<Particle> particles_per_human;
        for (int j = 0; j < N_particles; j++) {
            particles_per_human.push_back(Particle(&graph));
        }
        particles.push_back(particles_per_human);
    }
    particle_weights = std::vector<std::vector<double>>(
        N_humans_max, std::vector<double>(N_particles, 1.0 / N_particles));

    // init random number generator
    std::random_device rd;
    mt = std::mt19937(rd());
}

std::vector<std::vector<double>> ParticleTracker::add_observation(
    std::vector<pybind11::dict> robot_perceptions) {
    auto merged_perceptions = merge_perceptions(robot_perceptions);
    std::vector<Point> robot_positions = merged_perceptions.first;
    std::vector<pybind11::dict> perceived_humans = merged_perceptions.second;
    std::vector<std::function<int()>> hypothesis_generators =
        assign_perceived_humans_to_internal_humans(perceived_humans);

    // --- update particles for each human individually ---
    for (int i = 0; i < N_humans_max; i++) {
        for (int j = 0; j < N_particles; j++) {
            int perception_index = hypothesis_generators[i]();
            if (perception_index == -1) {
                particle_weights[i][j] *= particles[i][j].likelihood_no_perception(robot_positions);
            } else {
                Point perceived_pos = perceived_humans[perception_index]["position"].cast<Point>();
                double position_stddev =
                    perceived_humans[perception_index]["position_stddev"].cast<double>();
                double perceived_heading =
                    perceived_humans[perception_index]["heading"].cast<double>();
                double heading_stddev =
                    perceived_humans[perception_index]["heading_stddev"].cast<double>();
                particles[i][j] = generate_new_particle_from_perception(
                    perceived_pos, position_stddev, perceived_heading, heading_stddev);
                particle_weights[i][j] = 1.0 / N_particles;
            }
        }
        normalize_weights(i);

        // --- resample particles ---
        const double resample_threshold = 0.1 / static_cast<double>(N_particles);
        std::discrete_distribution<int> resample_distribution(particle_weights[i].begin(),
                                                              particle_weights[i].end());
        for (int j = 0; j < N_particles; j++) {
            if (particle_weights[i][j] < resample_threshold) {
                particles[i][j] = particles[i][resample_distribution(mt)];
            }
            particles[i][j] = Particle(particles[i][resample_distribution(mt)]);
        }
        normalize_weights(i);

        // // calculate effective sample size
        // double effective_sample_size =
        //     1.0 / std::accumulate(particle_weights[i].begin(), particle_weights[i].end(), 0.0,
        //                           [](double sum, double weight) { return sum + weight * weight;
        //                           });
        // std::cout << "Effective sample size: " << effective_sample_size
        //           << "; N_particles: " << N_particles << std::endl;
    }

    return calc_individual_edge_probabilities();
}

std::pair<std::vector<Point>, std::vector<pybind11::dict>> ParticleTracker::merge_perceptions(
    std::vector<pybind11::dict> robot_perceptions) {
    std::pair<std::vector<Point>, std::vector<pybind11::dict>> merged_perceptions;
    for (const auto& perception : robot_perceptions) {
        Point robot_position = perception["position"].cast<Point>();
        merged_perceptions.first.push_back(robot_position);
        auto perceived_humans = perception["perceived_humans"].cast<std::vector<pybind11::dict>>();
        std::vector<pybind11::dict> new_perceptions_this_robot;
        for (const auto& perceived_human : perceived_humans) {
            Point perceived_pos = perceived_human["position"].cast<Point>();
            double perceived_heading = perceived_human["heading"].cast<double>();
            double perceived_pos_variance =
                std::pow(XY_STDDEV * Agent::euclidean_distance(robot_position, perceived_pos), 2);
            double perceived_heading_variance = std::pow(HEADING_STDDEV, 2);

            bool matching_perception_found = false;
            for (auto& merged_human : merged_perceptions.second) {
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
        merged_perceptions.second.insert(merged_perceptions.second.end(),
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
    return Particle(belonging_edge, t, particles[0][0]);  // any particle as copy origin works
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

std::vector<std::vector<double>> ParticleTracker::predict() {
    for (int i = 0; i < N_humans_max; i++) {
        for (auto& particle : particles[i]) {
            particle.predict(T_step);
        }
    }
    return calc_individual_edge_probabilities();
}

void ParticleTracker::normalize_weights(int index_human) {
    double sum_weights = std::accumulate(particle_weights[index_human].begin(),
                                         particle_weights[index_human].end(), 0.0);
    for (int j = 0; j < N_particles; j++) {
        particle_weights[index_human][j] /= sum_weights;
        if (std::isnan(particle_weights[index_human][j])) {
            throw std::runtime_error("NaN detected in particle weights");
        }
    }
}

void ParticleTracker::print_weights(int index_human) {
    for (int i = 0; i < N_particles; i++) {
        std::cout << particle_weights[index_human][i] << " ";
    }
    std::cout << std::endl;
}

std::vector<std::vector<double>> ParticleTracker::calc_individual_edge_probabilities() {
    std::vector<std::vector<double>> individual_edge_probabilities;
    for (int i = 0; i < N_humans_max; i++) {
        std::vector<double> edge_probabilities_human_i(graph.edges.size(), 0.0);
        for (int j = 0; j < graph.edges.size(); j++) {
            for (int k = 0; k < N_particles; k++) {
                if (particles[i][k].is_human_on_edge(j)) {
                    edge_probabilities_human_i[j] += particle_weights[i][k];
                }
            }
        }
        individual_edge_probabilities.push_back(edge_probabilities_human_i);
    }
    return individual_edge_probabilities;
}

std::vector<std::function<int()>> ParticleTracker::assign_perceived_humans_to_internal_humans(
    std::vector<pybind11::dict> perceived_humans) {
    // --- allocate and fill c style cost matrix for hungarian algorithm lib ---
    int** cost_matrix = (int**)calloc(N_humans_max, sizeof(int*));
    for (int i = 0; i < N_humans_max; i++) {
        cost_matrix[i] = (int*)calloc(N_humans_max, sizeof(int));
        for (int j = 0; j < N_humans_max; j++) {
            cost_matrix[i][j] = 0;
        }
    }
    for (int i = 0; i < N_humans_max; i++) {  // tracks
        for (int j = 0; j < N_particles; j++) {
            for (int k = 0; k < N_humans_max; k++) {  // perceived humans
                if (k < perceived_humans.size()) {
                    auto particle = particles[i][j];
                    auto perceived_human = perceived_humans[k];
                    double graph_distance =
                        particle.assignment_cost(perceived_human["position"].cast<Point>(),
                                                 perceived_human["heading"].cast<double>());
                    cost_matrix[i][k] +=
                        static_cast<int>(1e4 * particle_weights[i][j] * graph_distance);
                } else {
                    cost_matrix[i][k] = 1e6;
                }
            }
        }
    }

    // --- assign perceived humans to internal humans using the hungarian algorithm ---
    hungarian_problem_t prob;
    int matrix_size = hungarian_init(&prob, cost_matrix, N_humans_max, N_humans_max,
                                     HUNGARIAN_MODE_MINIMIZE_COST);
    hungarian_solve(&prob);
    std::vector<std::function<int()>> perceived_human_per_internal_human(N_humans_max,
                                                                         []() { return -1; });
    for (int i = 0; i < N_humans_max; i++) {
        for (int j = 0; j < N_humans_max; j++) {
            if (prob.assignment[i][j] == 1) {
                if (j < perceived_humans.size()) {
                    perceived_human_per_internal_human[i] = [j]() { return j; };
                } else {
                    perceived_human_per_internal_human[j] = []() { return -1; };
                }
                break;
            }
        }
    }
    hungarian_free(&prob);
    for (int i = 0; i < N_humans_max; i++) {
        free(cost_matrix[i]);
    }
    return perceived_human_per_internal_human;
}