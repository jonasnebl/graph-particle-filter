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

Particle::Particle(graph_struct* graph_) : graph(graph_) {
    std::random_device rd;
    mt = std::mt19937(rd());
    edge = std::uniform_int_distribution<int>(0, graph->successor_edges.size() - 1)(mt);
    time_since_edge_change = 0.0;
    time_of_edge_change = get_random_time_of_edge_change(edge);
}

Particle::Particle(const Particle& p)
    : graph(p.graph),
      edge(p.edge),
      time_of_edge_change(p.time_of_edge_change),
      time_since_edge_change(p.time_since_edge_change) {
    std::random_device rd;
    mt = std::mt19937(rd());
    double t = p.time_since_edge_change / p.time_of_edge_change;
    time_of_edge_change = get_random_time_of_edge_change(edge);
    time_since_edge_change = t * time_of_edge_change;
}

Particle::Particle(int edge_, double t, graph_struct* graph_) : graph(graph_) {
    std::random_device rd;
    mt = std::mt19937(rd());
    edge = edge_;
    time_of_edge_change = get_random_time_of_edge_change(edge);
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
        if (Agent::check_viewline(robot_position, particle_position, graph->racks)) {
            double dist = Agent::euclidean_distance(robot_position, particle_position);
            likelihood *= 1 - Agent::probability_in_viewrange(dist);
        }
    }
    return likelihood;
}

void Particle::predict(double T_step) {
    time_since_edge_change += T_step;
    if (time_since_edge_change > time_of_edge_change) {  // switch to next edge
        edge = get_random_successor_edge(edge);
        time_since_edge_change = 0.0;
        time_of_edge_change = get_random_time_of_edge_change(edge);
        if (time_of_edge_change < 0.0) {
            throw std::runtime_error(std::string("Negative time of edge change: ") +
                                     std::to_string(time_of_edge_change));
        }
    }
}

int Particle::get_random_successor_edge(int current_edge) {
    int relative_edge_index = std::discrete_distribution<int>(
        graph->successor_edge_probabilities[current_edge].begin(),
        graph->successor_edge_probabilities[current_edge].end())(mt);
    return graph->successor_edges[current_edge][relative_edge_index];
}

double Particle::get_random_time_of_edge_change(int edge) {
    return std::weibull_distribution<double>(graph->duration_params[edge][0],
                                             graph->duration_params[edge][1])(mt);
}

bool Particle::is_human_on_edge(int edge_input) const { return edge == edge_input; }

double Particle::assignment_cost(Point position, double heading) {
    // assignment cost is a weighted sum of the cartesian distance and the heading difference
    return std::get<0>(
        ParticleTracker::edge_to_pose_distance_and_t(edge, position, heading, *graph));
}