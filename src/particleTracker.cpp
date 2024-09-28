#include "particleTracker.h"

#include <hungarian.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <vector>

#include "agent.h"
#include "particle.h"
#include "warehouse_data.h"

ParticleTracker::ParticleTracker(double T_step, int N_humans_max, int N_particles)
    : T_step(T_step), N_humans_max(N_humans_max), N_particles(N_particles) {
    // load graph
    nodes = warehouse_data::nodes;
    edges = warehouse_data::edges;
    edge_weights = warehouse_data::edge_weights;
    racks = warehouse_data::racks;

    // calculate successor edges
    for (int i = 0; i < edges.size(); i++) {
        std::vector<int> successor_edges_of_i;
        for (int j = 0; j < edges.size(); j++) {
            if (edges[i].second == edges[j].first) {
                successor_edges_of_i.push_back(j);
            }
        }
        successor_edges.push_back(successor_edges_of_i);
    }

    // load prediction model parameters
    try {
        std::ifstream file(pred_model_params_filename);
        nlohmann::json json_data;
        file >> json_data;
        pred_model_params = json_data.get<std::vector<std::vector<std::array<double, 3>>>>();
    } catch (const std::exception& e) {
        // initialize new prediction model parameters
        for (int i = 0; i < edges.size(); i++) {
            std::vector<std::array<double, 3>> params_per_edge;
            for (int j = 0; j < successor_edges[i].size(); j++) {
                double select_edge_j_probability;
                if (edges[successor_edges[i][j]].second ==
                    edges[i].first) {  // other direction edge
                    select_edge_j_probability = 0.1;
                } else {
                    select_edge_j_probability = (1 - 0.1) / (successor_edges[i].size() - 1);
                }
                params_per_edge.push_back(std::array<double, 3>(
                    {select_edge_j_probability, edge_weights[i] / HUMAN_VELOCITY_MEAN, 0.7}));
            }
            pred_model_params.push_back(params_per_edge);
        }
        save_pred_model_params();
    }

    // initialize particles
    for (int i = 0; i < N_humans_max; i++) {
        std::vector<Particle> particles_per_human;
        for (int j = 0; j < N_particles; j++) {
            particles_per_human.push_back(
                Particle(&nodes, &edges, &racks, &successor_edges, &pred_model_params));
        }
        particles.push_back(particles_per_human);
    }

    // init random number generator
    std::random_device rd;
    mt = std::mt19937(rd());
}

std::vector<double> ParticleTracker::add_observation(
    std::vector<pybind11::dict> robot_perceptions) {
    for (const auto& robot_perception : robot_perceptions) {
        add_single_observation(robot_perception);
    }
    return calculate_edge_probabilities();
}

void ParticleTracker::add_single_observation(pybind11::dict robot_perception) {
    Point robot_position = robot_perception["position"].cast<Point>();
    auto perceived_humans =
        robot_perception["perceived_humans"].cast<std::vector<pybind11::dict>>();

    std::vector<int> perceived_human_per_internal_human =
        assign_perceived_humans_to_internal_humans(robot_position, perceived_humans);

    // --- update particles for each human individually ---
    for (int i = 0; i < N_humans_max; i++) {
        int perception_index = perceived_human_per_internal_human[i];
        auto perceived_pos = perception_index == -1
                                 ? Point{std::nan(""), std::nan("")}
                                 : perceived_humans[perception_index]["position"].cast<Point>();
        auto perceived_heading = perception_index == -1
                                     ? std::nan("")
                                     : perceived_humans[perception_index]["heading"].cast<double>();

        // --- calculate distance of each particle ---
        std::vector<double> distances;
        for (auto& particle : particles[i]) {
            distances.push_back(
                particle.distance(robot_position, perceived_pos, perceived_heading));
        }
        update_particles(i, distances, perceived_pos, perceived_heading);
    }
}

std::vector<double> ParticleTracker::predict() {
    for (int i = 0; i < N_humans_max; i++) {
        for (auto& particle : particles[i]) {
            particle.predict(T_step);
        }
    }
    return calculate_edge_probabilities();
}

std::vector<double> ParticleTracker::calculate_edge_probabilities_one_human(int index_human) {
    std::vector<double> edge_probabilities(edges.size(), 0.0);
    for (int i = 0; i < edges.size(); i++) {
        for (const auto& particle : particles[index_human]) {
            if (particle.is_human_on_edge(i)) {
                edge_probabilities[i] += 1.0 / static_cast<double>(N_particles);
                break;
            }
        }
    }
    return edge_probabilities;
}

std::vector<double> ParticleTracker::calculate_edge_probabilities() {
    std::vector<double> edge_probabilities(edges.size(), 0.0);
    for (int i = 0; i < edges.size(); i++) {
        for (int j = 0; j < N_particles; j++) {
            for (int k = 0; k < N_humans_max; k++) {
                if (particles[k][j].is_human_on_edge(i)) {
                    edge_probabilities[i] += 1.0 / static_cast<double>(N_particles);
                    break;
                }
            }
        }
    }
    return edge_probabilities;
}

void ParticleTracker::save_pred_model_params() const {
    nlohmann::json json_data = pred_model_params;
    std::ofstream file(pred_model_params_filename);
    file << json_data.dump(4);
}

int ParticleTracker::get_belonging_edge(Point position, double heading) {
    std::vector<double> distances;
    for (int i = 0; i < edges.size(); i++) {
        Point p1 = nodes[edges[i].first];
        Point p2 = nodes[edges[i].second];
        double cartesian_distance_to_edge = distance_of_point_to_edge(position, p1, p2);
        ;
        double heading_difference =
            std::abs(heading - std::atan2(p2.second - p1.second, p2.first - p1.first));
        distances.push_back(cartesian_distance_to_edge + 2 * heading_difference);
    }
    return std::min_element(distances.begin(), distances.end()) - distances.begin();
}

double ParticleTracker::distance_of_point_to_edge(Point p, Point v, Point w) {
    const double l2 = std::hypot(v.first - w.first, v.second - w.second);      // Length of edge vw
    if (l2 == 0.0) return std::hypot(p.first - v.first, p.second - v.second);  // v == w case
    const double t = std::max(0.0, std::min(1.0, ((p.first - v.first) * (w.first - v.first) +
                                                  (p.second - v.second) * (w.second - v.second)) /
                                                     l2));
    const Point projection = {v.first + t * (w.first - v.first),
                              v.second + t * (w.second - v.second)};
    return std::hypot(p.first - projection.first, p.second - projection.second);
}

std::vector<int> ParticleTracker::assign_perceived_humans_to_internal_humans(
    Point robot_position, std::vector<pybind11::dict> perceived_humans) {
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
                    double manh_distance = Agent::manhattan_distance(
                        particle.get_position(), perceived_human["position"].cast<Point>());
                    double heading_distance = std::abs(particle.get_heading() -
                                                       perceived_human["heading"].cast<double>());
                    cost_matrix[j][k] += static_cast<int>(10000 / N_monte_carlo *
                                                          (manh_distance + 10 * heading_distance));                   
                } else {
                    cost_matrix[j][k] = 100000;
                }
            }
        }
    }

    // --- assign perceived humans to internal humans using the hungarian algorithm ---
    hungarian_problem_t prob;
    int matrix_size = hungarian_init(&prob, cost_matrix, N_humans_max, N_humans_max,
                                     HUNGARIAN_MODE_MINIMIZE_COST);
    hungarian_solve(&prob);
    std::vector<int> perceived_human_per_internal_human(N_humans_max, -1);
    for (int i = 0; i < N_humans_max; i++) {
        for (int j = 0; j < N_humans_max; j++) {
            if (prob.assignment[i][j] == 1) {
                if (j < perceived_humans.size()) {
                    perceived_human_per_internal_human[i] = j;
                } else {
                    perceived_human_per_internal_human[j] = -1;
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

void ParticleTracker::update_particles(int i, std::vector<double> distances, Point perceived_pos,
                                       double perceived_heading) {
    // --- Remove the particles with the highest distance and fill up with particles with lower
    std::vector<int> range(N_particles);
    std::iota(range.begin(), range.end(), 0);
    std::sort(range.begin(), range.end(),
              [&distances](int j1, int j2) { return distances[j1] < distances[j2]; });

    auto particles_copy = particles;
    int j = 0;
    // do {
    //     particles[i][j] = particles_copy[i][range[j]];
    //     j++;
    // } while(distances[range[j]] < 100 && j < N_particles-1);
    // int N_keep = j;
    int N_keep = 0.4 * N_particles;
    for (; j < N_keep; j++) {
        particles[i][j] = particles_copy[i][range[j]];
    }
    for (; j < N_particles; j++) {
        int random_particle = std::uniform_int_distribution<int>(0, N_keep - 1)(mt);
        particles[i][j] = Particle(particles[i][random_particle]);
    }
    if (!std::isnan(perceived_heading)) {
        int N_clever_resample = 0.1 * N_particles;
        for (j = N_particles - N_clever_resample; j < N_particles; j++) {
            int belonging_edge = get_belonging_edge(perceived_pos, perceived_heading);
            particles[i][j] = Particle(belonging_edge, particles[i][j]);
        }
    }
}

std::pair<int, int> ParticleTracker::find_max_element_index(
    const std::vector<std::vector<double>>& matrix) {
    int max_row = -1;
    int max_col = -1;
    double max_value = -1.0;
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = 0; j < matrix[i].size(); ++j) {
            if (matrix[i][j] > max_value) {
                max_value = matrix[i][j];
                max_row = i;
                max_col = j;
            }
        }
    }
    return std::make_pair(max_row, max_col);
}