#include "particle.h"

#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include "agent.h"
#include "particleTracker.h"
#include "simulation.h"

Particle::Particle(graph_struct* graph_,
                   std::vector<std::vector<std::array<double, 3>>>* pred_model_params_)
    : graph(graph_), pred_model_params(pred_model_params_) {
    std::random_device rd;
    mt = std::mt19937(rd());
    // edge = std::uniform_int_distribution<int>(0, successor_edges->size() - 1)(mt);
    for (int i = 0; i < graph->edges.size(); i++) {
        if (graph->edges[i].first == 24) {
            edge = i;
            break;
        }
    }
    next_edge = get_random_successor_edge(edge);
    time_since_edge_change = 0.8;
    time_of_edge_change = 1.0;
}

Particle::Particle(const Particle& p)
    : graph(p.graph),
      pred_model_params(p.pred_model_params),
      edge(p.edge),
      next_edge(p.next_edge),
      time_since_edge_change(p.time_since_edge_change) {
    std::random_device rd;
    mt = std::mt19937(rd());
    next_edge = get_random_successor_edge(edge);
    time_of_edge_change = get_random_time_of_edge_change(edge, next_edge);
}

Particle::Particle(int edge_, double t, const Particle& p)
    : graph(p.graph), pred_model_params(p.pred_model_params) {
    std::random_device rd;
    mt = std::mt19937(rd());
    edge = edge_;
    next_edge = get_random_successor_edge(edge);
    time_of_edge_change = get_random_time_of_edge_change(edge, next_edge);
    time_since_edge_change = t * time_of_edge_change;
}

Point Particle::get_position() {
    double x_edge_start = graph->nodes[graph->edges[edge].first].first;
    double x_edge_end = graph->nodes[graph->edges[edge].second].first;
    double y_edge_start = graph->nodes[graph->edges[edge].first].second;
    double y_edge_end = graph->nodes[graph->edges[edge].second].second;
    double x =
        x_edge_start + (x_edge_end - x_edge_start) * time_since_edge_change / time_of_edge_change;
    double y =
        y_edge_start + (y_edge_end - y_edge_start) * time_since_edge_change / time_of_edge_change;
    return std::make_pair(x, y);
}

double Particle::get_heading() {
    double x_edge_start = graph->nodes[graph->edges[edge].first].first;
    double x_edge_end = graph->nodes[graph->edges[edge].second].first;
    double y_edge_start = graph->nodes[graph->edges[edge].first].second;
    double y_edge_end = graph->nodes[graph->edges[edge].second].second;
    return std::atan2(y_edge_end - y_edge_start, x_edge_end - x_edge_start);
}

double Particle::likelihood_no_perception(std::vector<Point> robot_positions) {
    // Gives the probability of NOT seeing the human given particle state and robot position
    Point particle_position = get_position();
    double particle_heading = get_heading();
    double likelihood = 1.0;
    for (const auto& robot_position : robot_positions) {
        bool viewline = Agent::check_viewline(robot_position, particle_position, graph->racks);
        if (!viewline) {
            likelihood *= 1.0;
        } else {
            double dist = Agent::euclidean_distance(robot_position, particle_position);
            likelihood *= 1 - Agent::probability_in_viewrange(dist);
        }
    }
    return likelihood;
}

double Particle::measurement_noise_pdf(Point particle_position, Point measured_position) {
    double x_diff = particle_position.first - measured_position.first;
    double y_diff = particle_position.second - measured_position.second;
    double normalization_factor =
        1 / std::sqrt(std::pow(2 * 3.14159, 2) * std::pow(20 * XY_STDDEV, 4));
    double exponent =
        -0.5 * (std::pow(x_diff, 2) + std::pow(y_diff, 2)) / std::pow(20 * XY_STDDEV, 2);
    return normalization_factor * std::exp(exponent);
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
    return graph->successor_edges[current_edge][std::discrete_distribution(
        probabilities_for_next_node.begin(), probabilities_for_next_node.end())(mt)];
}

double Particle::get_random_time_of_edge_change(int current_edge, int next_edge) {
    std::vector<int> current_edge_successor_edges = graph->successor_edges[current_edge];
    int next_edge_relative_index =
        ParticleTracker::find_edge_relative_index(current_edge, next_edge, *graph);
    std::vector<std::array<double, 3>> weights = (*pred_model_params)[current_edge];
    return std::normal_distribution<double>(weights[next_edge_relative_index][1],
                                            weights[next_edge_relative_index][2])(mt);
}

bool Particle::is_human_on_edge(int edge_input) const { return edge == edge_input; }

double Particle::distance(Point position, double heading) {
    int measured_belonging_edge = Agent::get_belonging_edge(position, heading, *graph);
    double prob_distance = graph->prob_distance_matrix[edge][measured_belonging_edge];
    return 1 - prob_distance;
}