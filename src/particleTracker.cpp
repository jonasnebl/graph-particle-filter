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
    particle_weights = std::vector<std::vector<double>>(N_humans_max, std::vector<double>(N_particles, 1.0 / N_particles));

    // init random number generator
    std::random_device rd;
    mt = std::mt19937(rd());
}

std::vector<double> ParticleTracker::add_observation(std::vector<pybind11::dict> robot_perceptions) {
    auto merged_perceptions = merge_perceptions(robot_perceptions);
    std::vector<Point> robot_positions = merged_perceptions.first;
    std::vector<pybind11::dict> perceived_humans = merged_perceptions.second;

    std::vector<int> perceived_human_per_internal_human =
        assign_perceived_humans_to_internal_humans(robot_positions, perceived_humans);

    // --- update particles for each human individually ---
    for (int i = 0; i < N_humans_max; i++) {
        int perception_index = perceived_human_per_internal_human[i];

        if (perception_index == -1) {
            for (int j = 0; j < N_particles; j++) {
                particle_weights[i][j] *= particles[i][j].likelihood_no_perception(robot_position);
            }
            normalize_weights(i);
        } else {
            Point perceived_pos = perceived_humans[perception_index]["position"].cast<Point>();
            double perceived_heading =  perceived_humans[perception_index]["heading"].cast<double>();
            generate_particles_from_perception(i, robot_position, perceived_pos, perceived_heading);
            for (int j = 0; j < N_particles; j++) {
                particle_weights[i][j] = 1.0 / static_cast<double>(N_particles);
            }
        }
    }

    return calculate_edge_probabilities();
}

std::pair<std::vector<Point>, std::vector<pybind11::dict>> merge_perceptions(std::vector<pybind11::dict> robot_perceptions) {
    return;
}

void ParticleTracker::add_single_observation(pybind11::dict robot_perception) {
    Point robot_position = robot_perception["position"].cast<Point>();
    auto perceived_humans =
        robot_perception["perceived_humans"].cast<std::vector<pybind11::dict>>();

    
}

void ParticleTracker::generate_particles_from_perception(int i, Point robot_position, Point perceived_pos, double perceived_heading) {
    double dist = Agent::euclidean_distance(robot_position, perceived_pos);
    std::normal_distribution<double> position_noise(0, 5*XY_STDDEV * dist);
    std::normal_distribution<double> heading_noise(0, HEADING_STDDEV);
    for (int j = 0; j < N_particles; j++) {
        Point noisy_perceived_pos = perceived_pos;
        noisy_perceived_pos.first += position_noise(mt);
        noisy_perceived_pos.second += position_noise(mt);
        double noisy_perceived_heading = perceived_heading + heading_noise(mt);
        particles[i][j] = Particle(get_belonging_edge(noisy_perceived_pos, noisy_perceived_heading), particles[i][j]);
    }
}

int ParticleTracker::get_belonging_edge(Point position, double heading) {
    std::vector<double> distances;
    for (int i = 0; i < edges.size(); i++) {
        Point edge_start = nodes[edges[i].first];
        Point edge_end = nodes[edges[i].second];
        double cartesian_distance_to_edge = distance_of_point_to_edge(position, edge_start, edge_end);
        double head_distance = heading_distance(heading, std::atan2(edge_end.second - edge_start.second, edge_end.first - edge_start.first));
        distances.push_back(cartesian_distance_to_edge + 20 * head_distance);
    }
    return std::min_element(distances.begin(), distances.end()) - distances.begin();
}

double ParticleTracker::heading_distance(double heading1, double heading2) {
    double heading_difference = std::fmod(heading1 - heading2 + M_PI, 2 * M_PI);
    if (heading_difference < 0) {
        heading_difference += 2 * M_PI;
    }
    return std::abs(heading_difference - M_PI);
}

double ParticleTracker::distance_of_point_to_edge(Point point, Point edge_start, Point edge_end) {
    const double dx = edge_end.first - edge_start.first;
    const double dy = edge_end.second - edge_start.second;
    const double l2 = dx * dx + dy * dy; // squared length of the edge
    if (l2 == 0.0) { // edge_start == edge_end
        return std::hypot(point.first - edge_start.first, point.second - edge_start.second);
    }
    const double t = std::max(0.0, std::min(1.0, ((point.first - edge_start.first) * dx +
                                                  (point.second - edge_start.second) * dy) / l2));
    const Point projection = {edge_start.first + t * dx,
                              edge_start.second + t * dy};
    return std::hypot(point.first - projection.first, point.second - projection.second);
}

std::vector<double> ParticleTracker::predict() {
    for (int i = 0; i < N_humans_max; i++) {
        for (auto& particle : particles[i]) {
            particle.predict(T_step);
        }
    }
    return calculate_edge_probabilities();
}

void ParticleTracker::normalize_weights(int index_human) {
    double sum_weights = std::accumulate(particle_weights[index_human].begin(), particle_weights[index_human].end(), 0.0);
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

std::vector<double> ParticleTracker::calculate_edge_probabilities_one_human(int index_human) {
    std::vector<double> edge_probabilities(edges.size(), 0.0);
    for (int i = 0; i < edges.size(); i++) {
        for (int j = 0; j < N_particles; j++) {
            if (particles[index_human][j].is_human_on_edge(i)) {
                edge_probabilities[i] += particle_weights[index_human][j];
            }
        }
    }
    return edge_probabilities;
}

std::vector<double> ParticleTracker::calculate_edge_probabilities() {
    std::vector<std::vector<double>> edge_probabilities_for_every_internal_human;
    for (int i = 0; i < N_humans_max; i++) {
        edge_probabilities_for_every_internal_human.push_back(calculate_edge_probabilities_one_human(i));
    }
    std::vector<double> edge_probabilities(edges.size(), 1.0);
    for (int i = 0; i < edges.size(); i++) {
        for (int j = 0; j < N_humans_max; j++) {
            edge_probabilities[i] *= 1 - edge_probabilities_for_every_internal_human[j][i];
        }
        edge_probabilities[i] = 1 - edge_probabilities[i];
    }
    return edge_probabilities;
}

void ParticleTracker::save_pred_model_params() const {
    nlohmann::json json_data = pred_model_params;
    std::ofstream file(pred_model_params_filename);
    file << json_data.dump(4);
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
    const int N_monte_carlo = 500;
    std::uniform_int_distribution<int> get_random_particle(0, N_particles - 1);
    for (int i = 0; i < N_monte_carlo; i++) {
        for (int j = 0; j < N_humans_max; j++) {
            for (int k = 0; k < N_humans_max; k++) {
                if (k < perceived_humans.size()) {
                    auto particle = particles[j][get_random_particle(mt)];
                    auto perceived_human = perceived_humans[k];
                    double manh_distance = Agent::manhattan_distance(
                        particle.get_position(), perceived_human["position"].cast<Point>());
                    double head_distance = heading_distance(particle.get_heading(), perceived_human["heading"].cast<double>());
                    cost_matrix[j][k] += static_cast<int>(10000 / N_monte_carlo *
                                                          (manh_distance + 20 * head_distance));                   
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