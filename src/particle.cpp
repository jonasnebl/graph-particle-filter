#include "particle.h"

#include <array>
#include <cmath>
#include <random>
#include <tuple>
#include <vector>
#include <iostream>

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
}

std::vector<std::tuple<Point, double, double>> Particle::simulate_measurement(Point robot_position) {
    std::vector<std::tuple<Point, double, double>> measurements;
    for (auto& human : humans) {
        double distance = std::sqrt(std::pow(robot_position.first - (*nodes)[human.edge].first, 2) +
                                    std::pow(robot_position.second - (*nodes)[human.edge].second, 2));
        double xy_noise = 0.1 * distance;
        double velocity_noise = 0.1 * distance;
        double heading_noise = 0.1 * distance;

        double x_edge_start = (*nodes)[(*edges)[human.edge].first].first;
        double x_edge_end = (*nodes)[(*edges)[human.edge].second].first;
        double y_edge_start = (*nodes)[(*edges)[human.edge].first].second;
        double y_edge_end = (*nodes)[(*edges)[human.edge].second].second;

        // position
        double x = x_edge_start + (x_edge_end - x_edge_start) * human.time_of_edge_change / human.time_of_edge_change;
        double y = y_edge_start + (y_edge_end - y_edge_start) * human.time_of_edge_change / human.time_of_edge_change;

        // Debugging: Check if racks is null
        if (racks == nullptr) {
            std::cerr << "Error: racks pointer is null!" << std::endl;
            return measurements;
        }

        if (Agent::check_viewline(robot_position, std::make_pair(x, y), *racks)) {
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
            human.edge = human.next_edge;
            human.time_since_edge_change = 0;

            // determine next next edge
            std::vector<std::array<double, 3>> weights = (*pred_model_params)[human.edge];
            std::vector<double> probabilities_for_next_node;
            for (const auto& next_edge_weights : weights) {
                probabilities_for_next_node.push_back(next_edge_weights[0]);
            }
            human.next_edge =
                std::discrete_distribution(probabilities_for_next_node.begin(), probabilities_for_next_node.end())(mt);
            human.time_of_edge_change =
                std::normal_distribution<double>(weights[human.next_edge][1], weights[human.next_edge][2])(mt);
        }
    }
}