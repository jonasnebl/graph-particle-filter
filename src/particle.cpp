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
    // random initialization
    std::random_device rd;
    mt = std::mt19937(rd());
    edge = std::uniform_int_distribution<int>(0, graph->successor_edges.size() - 1)(mt);
    time_since_edge_change = 0.0;
    time_of_edge_change = get_random_time_of_edge_change(edge);
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
    double likelihood = 1.0;
    for (const auto& robot_position : robot_positions) {
        if (Agent::check_viewline(robot_position, particle_position, graph->racks)) {
            double dist = Agent::euclidean_distance(robot_position, particle_position);
            likelihood *= 1 - Agent::probability_in_viewrange(dist);
        }
    }
    return likelihood;
}

void Particle::rewrite_from_perception(Point perceived_pos, double position_stddev,
                                       double perceived_heading, double heading_stddev) {
    std::normal_distribution<double> position_noise(0, position_stddev);
    std::normal_distribution<double> heading_noise(0, heading_stddev);

    Point noisy_perceived_pos = perceived_pos;
    noisy_perceived_pos.first += position_noise(mt);
    noisy_perceived_pos.second += position_noise(mt);
    double noisy_perceived_heading = perceived_heading + heading_noise(mt);

    std::tuple<int, double> belonging_edge_and_t =
        get_belonging_edge(noisy_perceived_pos, noisy_perceived_heading, *graph);
    edge = std::get<0>(belonging_edge_and_t);
    double t = std::get<1>(belonging_edge_and_t);
    time_of_edge_change = get_random_time_of_edge_change(edge);
    time_since_edge_change = t * time_of_edge_change;
}

void Particle::rewrite_from_other_particle(const Particle& p) {
    edge = p.edge;
    // sample time of edge change new to generate more variability in the particles
    double t = p.time_since_edge_change / p.time_of_edge_change;
    time_of_edge_change = get_random_time_of_edge_change(edge);
    time_since_edge_change = t * time_of_edge_change;
}

std::tuple<int, double> Particle::get_belonging_edge(Point position, double heading,
                                                     const graph_struct& graph) {
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

std::tuple<double, double> Particle::edge_to_pose_distance_and_t(int edge, Point position,
                                                                 double heading,
                                                                 const graph_struct& graph) {
    const Point edge_start = graph.nodes[graph.edges[edge].first];
    const Point edge_end = graph.nodes[graph.edges[edge].second];

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
    double heading_dist = ParticleTracker::heading_distance(heading, edge_heading);

    // --- return weighted sum of cartesian distance and heading difference ---
    return std::make_tuple(cartesian_distance + HEADING_WEIGHT * heading_dist, t);
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
    return Agent::euclidean_distance(get_position(), position) +
           HEADING_WEIGHT * ParticleTracker::heading_distance(heading, get_heading());
}