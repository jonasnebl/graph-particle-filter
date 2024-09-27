#include "particle.h"

#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include "agent.h"

Particle::Particle(std::vector<Point>* nodes, std::vector<std::pair<int, int>>* edges,
                   std::vector<std::vector<Point>>* racks,
                   std::vector<std::vector<int>>* successor_edges,
                   std::vector<std::vector<std::array<double, 3>>>* pred_model_params)
    : nodes(nodes),
      edges(edges),
      racks(racks),
      successor_edges(successor_edges),
      pred_model_params(pred_model_params) {
    std::random_device rd;
    mt = std::mt19937(rd());
    edge = std::uniform_int_distribution<int>(0, successor_edges->size() - 1)(mt);
    next_edge = get_random_successor_edge(edge);
    time_since_edge_change = 0.8;
    time_of_edge_change = 1.0;
}

Particle::Particle(const Particle& p)
    : nodes(p.nodes),
      edges(p.edges),
      racks(p.racks),
      successor_edges(p.successor_edges),
      pred_model_params(p.pred_model_params),
      edge(p.edge),
      next_edge(p.next_edge),
      time_since_edge_change(p.time_since_edge_change) {
    std::random_device rd;
    mt = std::mt19937(rd());
    next_edge = get_random_successor_edge(edge);
    time_of_edge_change = get_random_time_of_edge_change(edge, next_edge);
}

Particle::Particle(int edge_, const Particle& p)
    : nodes(p.nodes),
      edges(p.edges),
      racks(p.racks),
      successor_edges(p.successor_edges),
      pred_model_params(p.pred_model_params) {
    std::random_device rd;
    mt = std::mt19937(rd());
    edge = edge_;
    next_edge = get_random_successor_edge(edge);
    time_of_edge_change = get_random_time_of_edge_change(edge, next_edge);
    time_since_edge_change = 0.5 * get_random_time_of_edge_change(edge, next_edge);
}

Point Particle::get_position() {
    double x_edge_start = (*nodes)[(*edges)[edge].first].first;
    double x_edge_end = (*nodes)[(*edges)[edge].second].first;
    double y_edge_start = (*nodes)[(*edges)[edge].first].second;
    double y_edge_end = (*nodes)[(*edges)[edge].second].second;
    double x =
        x_edge_start + (x_edge_end - x_edge_start) * time_since_edge_change / time_of_edge_change;
    double y =
        y_edge_start + (y_edge_end - y_edge_start) * time_since_edge_change / time_of_edge_change;
    return std::make_pair(x, y);
}

double Particle::get_heading() {
    double x_edge_start = (*nodes)[(*edges)[edge].first].first;
    double x_edge_end = (*nodes)[(*edges)[edge].second].first;
    double y_edge_start = (*nodes)[(*edges)[edge].first].second;
    double y_edge_end = (*nodes)[(*edges)[edge].second].second;
    return std::atan2(y_edge_end - y_edge_start, x_edge_end - x_edge_start);
}

double Particle::distance(Point robot_position, Point measured_position, double heading) {
    Point ego_position = get_position();
    double edge_heading = get_heading();

    double distance = 0;
    bool human_should_be_visible = Agent::check_viewline(robot_position, ego_position, *racks);
    if (std::isnan(heading) && !human_should_be_visible) {
        return 0;
    } else if (std::isnan(heading) && human_should_be_visible) {
        previous_distance += 10;
        return previous_distance;
    } else if (!std::isnan(heading) && !human_should_be_visible) {
        previous_distance += 10;
        return previous_distance;
    } else if (!std::isnan(heading) && human_should_be_visible) {
        distance = (std::pow(measured_position.first - ego_position.first, 2) +
                    std::pow(measured_position.second - ego_position.second, 2)) +
                   100 * std::pow(std::abs(heading - edge_heading), 2);
        previous_distance = distance;
        return distance;
    }
    return 0;
}

void Particle::predict(double T_step) {
    time_since_edge_change += T_step;
    if (time_since_edge_change > time_of_edge_change) {
        // switch to next edge
        edge = next_edge;
        time_since_edge_change = 0;

        // determine next next edge
        next_edge = get_random_successor_edge(edge);
        time_of_edge_change = get_random_time_of_edge_change(edge, next_edge);
    }
}

int Particle::get_random_successor_edge(int current_edge) {
    std::vector<double> probabilities_for_next_node;
    for (const auto& next_edge_weights : (*pred_model_params)[current_edge]) {
        probabilities_for_next_node.push_back(next_edge_weights[0]);
    }
    return (*successor_edges)[current_edge][std::discrete_distribution(
        probabilities_for_next_node.begin(), probabilities_for_next_node.end())(mt)];
}

double Particle::get_random_time_of_edge_change(int current_edge, int next_edge) {
    std::vector<int> current_edge_successor_edges = (*successor_edges)[current_edge];
    int next_edge_relative_index = find_edge_relative_index(current_edge, next_edge);
    std::vector<std::array<double, 3>> weights = (*pred_model_params)[current_edge];
    return std::normal_distribution<double>(weights[next_edge_relative_index][1],
                                            weights[next_edge_relative_index][2])(mt);
}

int Particle::find_edge_relative_index(int edge, int next_edge) const {
    return find((*successor_edges)[edge].begin(), (*successor_edges)[edge].end(), next_edge) -
           (*successor_edges)[edge].begin();
}

bool Particle::is_human_on_edge(int edge_input) const { return edge == edge_input; }