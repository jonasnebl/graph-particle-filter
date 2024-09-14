#include "particle.h"

#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include "agent.h"

Particle::Particle(int N_humans_max, std::vector<Point>* nodes, std::vector<std::pair<int, int>>* edges,
                   std::vector<std::vector<Point>>* racks, std::vector<std::vector<int>>* successor_edges,
                   std::vector<std::vector<std::array<double, 3>>>* pred_model_params)
    : nodes(nodes), edges(edges), racks(racks), successor_edges(successor_edges), pred_model_params(pred_model_params) {
    std::random_device rd;
    mt = std::mt19937(rd());
    auto dist = std::uniform_int_distribution<int>(0, successor_edges->size() - 1);
    for (int i = 0; i < N_humans_max; i++) {
        Human h;
        h.edge = dist(mt);
        h.next_edge = get_random_successor_edge(h.edge);
        h.previous_edge = -1;
        h.time_since_edge_change = 1.0;
        h.time_of_edge_change = 1.0;
        humans.push_back(h);
    }
}

Particle::Particle(const Particle& p)
    : nodes(p.nodes),
      edges(p.edges),
      racks(p.racks),
      successor_edges(p.successor_edges),
      pred_model_params(p.pred_model_params) {
    std::random_device rd;
    mt = std::mt19937(rd());
    humans = p.humans;
    for (auto& human : humans) {
        human.next_edge = get_random_successor_edge(human.edge);
        human.time_of_edge_change = get_random_time_of_edge_change(human.edge, human.next_edge);
    }
}

std::vector<std::tuple<Point, double, double>> Particle::simulate_measurement(Point robot_position) {
    std::vector<std::tuple<Point, double, double>> measurements;
    for (auto& human : humans) {
        double x_edge_start = (*nodes)[(*edges)[human.edge].first].first;
        double x_edge_end = (*nodes)[(*edges)[human.edge].second].first;
        double y_edge_start = (*nodes)[(*edges)[human.edge].first].second;
        double y_edge_end = (*nodes)[(*edges)[human.edge].second].second;

        // position
        double x =
            x_edge_start + (x_edge_end - x_edge_start) * human.time_since_edge_change / human.time_of_edge_change;
        double y =
            y_edge_start + (y_edge_end - y_edge_start) * human.time_since_edge_change / human.time_of_edge_change;

        if (Agent::check_viewline(robot_position, std::make_pair(x, y), *racks)) {
            // determine noise variance
            double distance_to_robot =
                std::sqrt(std::pow(robot_position.first - x, 2) + std::pow(robot_position.second - y, 2));
            double xy_noise = 0.001 * distance_to_robot;
            double velocity_noise = 0.001 * distance_to_robot;
            double heading_noise = 0.001 * distance_to_robot;

            x += std::normal_distribution<double>(0, xy_noise)(mt);
            y += std::normal_distribution<double>(0, xy_noise)(mt);
            Point measured_position = std::make_pair(x, y);

            // heading
            double heading = std::atan2(y_edge_end - y_edge_start, x_edge_end - x_edge_start);
            heading += std::normal_distribution<double>(0, heading_noise)(mt);

            // velocity
            double edge_length =
                std::sqrt(std::pow(x_edge_end - x_edge_start, 2) + std::pow(y_edge_end - y_edge_start, 2));
            double velocity = edge_length / human.time_of_edge_change;
            velocity += std::normal_distribution<double>(0, velocity_noise)(mt);
            measurements.push_back(std::make_tuple(measured_position, heading, velocity));
        }
    }
    return measurements;
}

void Particle::predict(double T_step) {
    for (auto& human : humans) {
        human.time_since_edge_change += T_step;
        if (human.time_since_edge_change > human.time_of_edge_change) {
            // switch to next edge
            human.previous_edge = human.edge;
            human.edge = human.next_edge;
            human.time_since_edge_change = 0;

            // determine next next edge
            human.next_edge = get_random_successor_edge(human.edge);
            human.time_of_edge_change = get_random_time_of_edge_change(human.edge, human.next_edge);
        }
    }
}

int Particle::get_random_successor_edge(int current_edge) {
    std::vector<double> probabilities_for_next_node;
    for (const auto& next_edge_weights : (*pred_model_params)[current_edge]) {
        probabilities_for_next_node.push_back(next_edge_weights[0]);
    }
    return (*successor_edges)[current_edge][std::discrete_distribution(probabilities_for_next_node.begin(),
                                                                       probabilities_for_next_node.end())(mt)];
}

double Particle::get_random_time_of_edge_change(int current_edge, int next_edge) {
    std::vector<int> current_edge_successor_edges = (*successor_edges)[current_edge];
    int next_edge_relative_index = find_edge_relative_index(current_edge, next_edge);
    std::vector<std::array<double, 3>> weights = (*pred_model_params)[current_edge];
    return std::exp(std::normal_distribution<double>(weights[next_edge_relative_index][1],
                                            weights[next_edge_relative_index][2])(mt));
}

void Particle::update_params(double distance_metric) {
    const double learning_rate = 5e-5;
    for (auto& human : humans) {
        if (human.previous_edge != -1) {
            int edge_relative_index = find_edge_relative_index(human.previous_edge, human.edge);
            if (distance_metric < 1000) {      
                // adapt next edge probability
                double selected_node_probability_addition =
                    learning_rate * (1 - (*pred_model_params)[human.previous_edge][edge_relative_index][0]);
                (*pred_model_params)[human.previous_edge][edge_relative_index][0] += selected_node_probability_addition;
                for (int i = 0; i < (*pred_model_params)[human.previous_edge].size(); i++) {
                    if (i != edge_relative_index) {
                        (*pred_model_params)[human.previous_edge][i][0] -=
                            selected_node_probability_addition * (*pred_model_params)[human.previous_edge][i][0] / (1 - (*pred_model_params)[human.previous_edge][edge_relative_index][0]);
                    }
                }
                // adapt time of edge change
                double time_of_edge_change_addition =
                    10 * learning_rate *
                    ((*pred_model_params)[human.previous_edge][edge_relative_index][1] - human.time_of_edge_change);
                (*pred_model_params)[human.previous_edge][edge_relative_index][1] -= time_of_edge_change_addition;
            } else {
                double selected_node_probability_subtraction =
                    learning_rate * (*pred_model_params)[human.previous_edge][edge_relative_index][0];
                (*pred_model_params)[human.previous_edge][edge_relative_index][0] -= selected_node_probability_subtraction;
                for (int i = 0; i < (*pred_model_params)[human.previous_edge].size(); i++) {
                    if (i != edge_relative_index) {
                        (*pred_model_params)[human.previous_edge][i][0] +=
                            selected_node_probability_subtraction * (1 - (*pred_model_params)[human.previous_edge][i][0]) / (*pred_model_params)[human.previous_edge][edge_relative_index][0];
                    }
                }
            }
        }
    }
}

int Particle::find_edge_relative_index(int edge, int next_edge) const {
    return find((*successor_edges)[edge].begin(), (*successor_edges)[edge].end(), next_edge) -
           (*successor_edges)[edge].begin();
}