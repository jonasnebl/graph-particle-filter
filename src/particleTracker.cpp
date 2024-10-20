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
    // load prediction model parameters
    try {
        std::ifstream file(pred_model_params_filename);
        nlohmann::json json_data;
        file >> json_data;
        pred_model_params = json_data.get<std::vector<std::vector<std::array<double, 3>>>>();
    } catch (const std::exception& e) {
        std::cout << "No prediction model parameters found." << std::endl;
    }

    // prob_distance_matrix is based on prediciton model parameters
    graph.prob_distance_matrix = calc_prob_distance_matrix();

    // init particles
    for (int i = 0; i < N_humans_max; i++) {
        std::vector<Particle> particles_per_human;
        for (int j = 0; j < N_particles; j++) {
            particles_per_human.push_back(Particle(&graph, &pred_model_params));
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

        // // --- resample particles ---
        // const double resample_threshold = 0.5 / static_cast<double>(N_particles);
        // std::discrete_distribution<int> resample_distribution(particle_weights[i].begin(),
        //                                                       particle_weights[i].end());
        // for (int j = 0; j < N_particles; j++) {
        //     if (particle_weights[i][j] < resample_threshold) {
        //         particles[i][j] = particles[i][resample_distribution(mt)];
        //     }
        //     particles[i][j] = Particle(particles[i][resample_distribution(mt)]);
        // }
        // normalize_weights(i);

        // calculate effective sample size
        double effective_sample_size =
            1.0 / std::accumulate(particle_weights[i].begin(), particle_weights[i].end(), 0.0,
                                  [](double sum, double weight) { return sum + weight * weight; });
        std::cout << "Effective sample size: " << effective_sample_size
                  << "; N_particles: " << N_particles << std::endl;
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
        Point edge_start = graph.nodes[graph.edges[i].first];
        Point edge_end = graph.nodes[graph.edges[i].second];
        auto cartesian_distance_to_edge_and_t =
            distance_of_point_to_edge(position, edge_start, edge_end);
        double cartesian_distance_to_edge = std::get<0>(cartesian_distance_to_edge_and_t);
        t_values.push_back(std::get<1>(cartesian_distance_to_edge_and_t));
        double head_distance = heading_distance(
            heading,
            std::atan2(edge_end.second - edge_start.second, edge_end.first - edge_start.first));
        distances.push_back(cartesian_distance_to_edge + HEADING_WEIGHT * head_distance);
    }
    int min_index = std::min_element(distances.begin(), distances.end()) - distances.begin();
    return std::make_tuple(min_index, t_values[min_index]);
}

double ParticleTracker::heading_distance(double heading1, double heading2) {
    double heading_difference = std::fmod(heading1 - heading2 + M_PI, 2 * M_PI);
    if (heading_difference < 0) {
        heading_difference += 2 * M_PI;
    }
    return std::abs(heading_difference - M_PI);
}

std::tuple<double, double> ParticleTracker::distance_of_point_to_edge(Point point, Point edge_start,
                                                                      Point edge_end) {
    const double dx = edge_end.first - edge_start.first;
    const double dy = edge_end.second - edge_start.second;
    const double l2 = dx * dx + dy * dy;  // squared length of the edge
    if (l2 == 0.0) {                      // edge_start == edge_end
        return std::make_tuple(
            std::hypot(point.first - edge_start.first, point.second - edge_start.second), 0.0);
    }
    const double t = std::max(0.0, std::min(1.0, ((point.first - edge_start.first) * dx +
                                                  (point.second - edge_start.second) * dy) /
                                                     l2));
    const Point projection = {edge_start.first + t * dx, edge_start.second + t * dy};
    return std::make_tuple(
        std::hypot(point.first - projection.first, point.second - projection.second), t);
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

std::vector<double> ParticleTracker::calc_edge_probabilities_one_human(int index_human) {
    std::vector<double> edge_probabilities(graph.edges.size(), 0.0);
    for (int i = 0; i < graph.edges.size(); i++) {
        for (int j = 0; j < N_particles; j++) {
            if (particles[index_human][j].is_human_on_edge(i)) {
                edge_probabilities[i] += particle_weights[index_human][j];
            }
        }
    }
    return edge_probabilities;
}

std::vector<std::vector<double>> ParticleTracker::calc_individual_edge_probabilities() {
    std::vector<std::vector<double>> individual_edge_probabilities;
    for (int i = 0; i < N_humans_max; i++) {
        individual_edge_probabilities.push_back(calc_edge_probabilities_one_human(i));
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
    const int N_monte_carlo = 100;
    std::uniform_int_distribution<int> get_random_particle(0, N_particles - 1);
    for (int i = 0; i < N_monte_carlo; i++) {
        for (int j = 0; j < N_humans_max; j++) {
            for (int k = 0; k < N_humans_max; k++) {
                if (k < perceived_humans.size()) {
                    auto particle = particles[j][get_random_particle(mt)];
                    auto perceived_human = perceived_humans[k];
                    double graph_distance =
                        particle.assignment_cost(perceived_human["position"].cast<Point>(),
                                                 perceived_human["heading"].cast<double>());
                    cost_matrix[j][k] += static_cast<int>(1e4 * graph_distance);
                } else {
                    cost_matrix[j][k] = 1e8;
                }
            }
        }
    }

    // --- assign perceived humans to internal humans using the hungarian algorithm ---
    hungarian_problem_t prob;
    int matrix_size = hungarian_init(&prob, cost_matrix, N_humans_max, N_humans_max,
                                     HUNGARIAN_MODE_MINIMIZE_COST);
    hungarian_print_costmatrix(&prob);
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

std::vector<std::vector<double>> ParticleTracker::calc_prob_distance_matrix() {
    std::vector<std::vector<double>> prob_distance_matrix(
        graph.edges.size(), std::vector<double>(graph.edges.size(), 1.0));
    for (int i = 0; i < graph.edges.size(); i++) {
        for (int j = 0; j < graph.edges.size(); j++) {
            if (i == j) {
                prob_distance_matrix[i][j] = 1.0;
            } else {
                std::vector<int> edge_path;
                if (graph.edges[i].second == graph.edges[j].first) {
                    edge_path.push_back(i);
                    edge_path.push_back(j);
                } else {
                    edge_path.push_back(i);
                    std::vector<int> node_path =
                        Simulation::dijkstra(graph.edges[i].second, graph.edges[j].first, graph);
                    for (int k = 0; k < node_path.size() - 1; k++) {
                        for (int l = 0; l < graph.edges.size(); l++) {
                            if (graph.edges[l].first == node_path[k] &&
                                graph.edges[l].second == node_path[k + 1]) {
                                edge_path.push_back(l);
                                break;
                            }
                        }
                    }
                    edge_path.push_back(j);
                }
                for (int k = 0; k < edge_path.size() - 1; k++) {
                    prob_distance_matrix[i][j] *=
                        pred_model_params[edge_path[k]][find_edge_relative_index(
                            edge_path[k], edge_path[k + 1], graph)][0];
                }
            }
        }
    }
    // // print prob_distance_matrix
    // for (int i = 0; i < 10; i++) {
    //     for (int j = 0; j < 10; j++) {
    //         std::cout << prob_distance_matrix[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    return prob_distance_matrix;
}

int ParticleTracker::find_edge_relative_index(int edge, int next_edge, graph_struct& graph) {
    return std::find(graph.successor_edges[edge].begin(), graph.successor_edges[edge].end(),
                     next_edge) -
           graph.successor_edges[edge].begin();
}