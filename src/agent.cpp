#include "agent.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include "particleTracker.h"
#include "simulation.h"

Agent::Agent(double T_step, AgentType type_, Simulation* simulation_)
    : _T_step(T_step), type(type_), simulation(simulation_) {
    std::random_device rd;
    mt = std::mt19937(rd());
    if (type == AgentType::HUMAN) {
        velocity_distribution =
            std::normal_distribution<double>(HUMAN_VELOCITY_MEAN, HUMAN_VELOCITY_STDDEV);
    } else {
        velocity_distribution = std::normal_distribution<double>(ROBOT_VELOCITY, 0.0);
    }
    position_noise = std::normal_distribution<double>(0, XY_STDDEV);
    heading_noise = std::normal_distribution<double>(0, HEADING_STDDEV);

    int start_node_index = random_staging_node();
    position = simulation->graph.nodes[start_node_index];
    path = std::deque<std::pair<Point, double>>();
    add_node_to_deque(start_node_index, get_random_velocity());
    current_final_path_node = start_node_index;
}

void Agent::step() {
    while (path.size() <= 10) {  // keep path length at least at 10
        add_new_double_cycle_to_deque();
    }
    double x_target = path.front().first.first;
    double y_target = path.front().first.second;
    velocity = path.front().second;

    double dist_x_to_target = x_target - position.first;
    double dist_y_to_target = y_target - position.second;
    heading = std::atan2(dist_y_to_target, dist_x_to_target);
    position.first += velocity * std::cos(heading) * _T_step;
    position.second += velocity * std::sin(heading) * _T_step;

    double dist_to_target = euclidean_distance(position, path.front().first);
    double dist_covered_in_one_step = velocity * _T_step;
    if (dist_covered_in_one_step > dist_to_target) {  // target reached --> next target
        path.pop_front();
    }
    return;
}

pybind11::dict Agent::log_state() {
    pybind11::dict state;
    state["position"] = position;
    state["belonging_edge"] = get_belonging_edge(position, heading, simulation->graph);
    if (type == AgentType::HUMAN) {
        state["type"] = "human";
    } else {
        state["type"] = "robot";
        state["perceived_humans"] = perceive_humans();
    }
    return state;
}

int Agent::get_belonging_edge(Point position, double heading, graph_struct& graph) {
    std::vector<double> distances;
    for (int i = 0; i < graph.edges.size(); i++) {
        Point p1 = graph.nodes[graph.edges[i].first];
        Point p2 = graph.nodes[graph.edges[i].second];
        double cartesian_distance_to_edge = std::get<0>(
            ParticleTracker::distance_of_point_to_edge(position, p1, p2));  // we don't need t
        double heading_difference = ParticleTracker::heading_distance(
            heading, std::atan2(p2.second - p1.second, p2.first - p1.first));
        distances.push_back(cartesian_distance_to_edge +
                            ParticleTracker::HEADING_WEIGHT * heading_difference);
    }
    return std::min_element(distances.begin(), distances.end()) - distances.begin();
}

bool Agent::check_viewline(Point pos1, Point pos2, std::vector<std::vector<Point>> racks) {
    for (const auto& polygon : racks) {
        for (size_t i = 0; i < polygon.size(); ++i) {
            Point p1 = polygon[i];
            Point p2 = polygon[(i + 1) % polygon.size()];
            if (do_intersect(pos1, pos2, p1, p2)) {
                return false;  // Obstruction found
            }
        }
    }
    return true;
}

double Agent::get_random_velocity() { return velocity_distribution(mt); }

std::vector<pybind11::dict> Agent::perceive_humans() {
    std::vector<pybind11::dict> result;
    for (int i = 0; i < simulation->_N_humans + simulation->_N_robots; i++) {
        if (simulation->agents[i].type == AgentType::HUMAN) {
            Agent human = simulation->agents[i];
            if (check_viewline(position, human.position, simulation->graph.racks)) {
                if (random_check_viewrange(position, human.position)) {
                    double dist = euclidean_distance(position, human.position);
                    pybind11::dict perceived_human;
                    auto noisy_position = human.position;
                    noisy_position.first += position_noise(mt) * dist;
                    noisy_position.second += position_noise(mt) * dist;
                    perceived_human["position"] = noisy_position;
                    perceived_human["heading"] = human.heading + heading_noise(mt);
                    result.push_back(perceived_human);
                }
            }
        }
    }
    return result;
}

void Agent::add_new_double_cycle_to_deque() {
    // double cycles are simulated
    // one double cycle consists of legs:
    // 1. random node in staging -> random node in storage
    // 2. random node in storage -> another random node in storage
    // 3. another random node in storage -> random node in storage

    int path_length_before_new_double_cycle = path.size();

    if (random_leave_warehouse() &&
        type == AgentType::HUMAN) {  // only humans can leave the warehouse
        // --- 1. leg ---
        int target_exit_node = simulation->graph.exit_nodes[0];
        std::vector<int> path_to_target =
            simulation->dijkstra(current_final_path_node, target_exit_node, simulation->graph);
        add_path_to_deque(path_to_target, get_random_velocity());
        add_node_to_deque(target_exit_node, OUT_OF_WAREHOUSE_VELOCITY);  // generates a pause
        // --- 2. leg ---
        int target_staging_node = random_staging_node();
        path_to_target =
            simulation->dijkstra(target_exit_node, target_staging_node, simulation->graph);
        add_path_to_deque(path_to_target, get_random_velocity());
        add_node_to_deque(target_staging_node, PAUSE_VELOCITY);  // generates a pause
        current_final_path_node = target_exit_node;
    } else {
        // --- 1. leg ---
        int target_node_1 = random_storage_node();
        std::vector<int> path_to_target =
            simulation->dijkstra(current_final_path_node, target_node_1, simulation->graph);
        add_path_to_deque(path_to_target, get_random_velocity());
        add_node_to_deque(target_node_1, PAUSE_VELOCITY);  // generates a pause
        // --- 2. leg ---
        int target_node_2 = random_storage_node();
        path_to_target = simulation->dijkstra(target_node_1, target_node_2, simulation->graph);
        add_path_to_deque(path_to_target, get_random_velocity());
        add_node_to_deque(target_node_2, PAUSE_VELOCITY);  // generates a pause
        // --- 3. leg ---
        int target_staging_node = random_staging_node();
        path_to_target =
            simulation->dijkstra(target_node_2, target_staging_node, simulation->graph);
        add_path_to_deque(path_to_target, get_random_velocity());
        add_node_to_deque(target_staging_node, PAUSE_VELOCITY);  // generates a pause
        current_final_path_node = target_staging_node;
    }
    // --- smooth ---
    for (int i = 0; i < SMOOTHING_ITERATIONS; i++) {  // smooth multiple times for better results
        smooth_path(path_length_before_new_double_cycle, path.size() - 1, SMOOTHING_STRENGTH);
    }
    return;
}

void Agent::add_path_to_deque(std::vector<int> path_to_target, double path_velocity) {
    for (int i = 1; i < path_to_target.size(); i++) {  // exclude first element
        add_node_to_deque(path_to_target[i], path_velocity);
    }
}

void Agent::add_node_to_deque(int node_index, double path_velocity) {
    double node_x =
        (simulation->graph.nodes)[node_index].first + simulation->get_trajectory_xy_noise();
    double node_y =
        simulation->graph.nodes[node_index].second + simulation->get_trajectory_xy_noise();
    path.push_back(std::make_pair(std::make_pair(node_x, node_y), path_velocity));
}

void Agent::smooth_path(int start, int end, double strength) {
    for (int i = start + 1; i < end - 1; i++) {
        double x_before = path[i - 1].first.first;
        double y_before = path[i - 1].first.second;
        double x_after = path[i + 1].first.first;
        double y_after = path[i + 1].first.second;

        double x_current = path[i].first.first;
        double y_current = path[i].first.second;

        // Calculate the projection of the current point onto the line segment
        double dx = x_after - x_before;
        double dy = y_after - y_before;
        double length_squared = dx * dx + dy * dy;
        double t = ((x_current - x_before) * dx + (y_current - y_before) * dy) / length_squared;

        // Clamp t to the range [0, 1] to ensure the projection is on the segment
        t = std::max(0.0, std::min(1.0, t));

        double x_projection = x_before + t * dx;
        double y_projection = y_before + t * dy;

        // Calculate the distance to the projection
        double distance = std::hypot(x_current - x_projection, y_current - y_projection);

        // Move the current point closer to the projection
        double move_x = (x_projection - x_current) * strength;
        double move_y = (y_projection - y_current) * strength;

        path[i].first.first += move_x;
        path[i].first.second += move_y;
    }
}

int Agent::random_staging_node() {
    std::uniform_int_distribution<int> staging_node_distribution(
        0, simulation->graph.staging_nodes.size() - 1);
    return (simulation->graph.staging_nodes)[staging_node_distribution(mt)];
}

int Agent::random_storage_node() {
    std::uniform_int_distribution<int> storage_node_distribution(
        0, simulation->graph.storage_nodes.size() - 1);
    return (simulation->graph.storage_nodes)[storage_node_distribution(mt)];
}

bool Agent::random_leave_warehouse() {
    std::uniform_real_distribution<double> unif(0, 1);
    return unif(mt) < LEAVE_WAREHOUSE_PROBABILITY;
}

double Agent::euclidean_distance(Point p1, Point p2) {
    return std::sqrt(std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2));
}

double Agent::manhattan_distance(Point p1, Point p2) {
    return std::abs(p1.first - p2.first) + std::abs(p1.second - p2.second);
}

// Function to check if two line segments intersect
bool Agent::do_intersect(Point p1, Point q1, Point p2, Point q2) {
    auto orientation = [](Point p, Point q, Point r) {
        double val = (q.second - p.second) * (r.first - q.first) -
                     (q.first - p.first) * (r.second - q.second);
        if (val == 0) return 0;    // collinear
        return (val > 0) ? 1 : 2;  // clock or counterclock wise
    };

    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    if (o1 != o2 && o3 != o4) return true;

    return false;
}

bool Agent::is_point_in_polygon(Point point, std::vector<Point> polygon) {
    int n = polygon.size();
    if (n < 3) return false;  // A polygon must have at least 3 vertices
    Point infinity_point = std::make_pair(point.first + 1e10, point.second);
    int intersection_count = 0;
    for (int i = 0; i < n; ++i) {
        Point p1 = polygon[i];
        Point p2 = polygon[(i + 1) % n];

        if (do_intersect(point, infinity_point, p1, p2)) {
            intersection_count++;
        }
    }
    return (intersection_count % 2 == 1);
}

bool Agent::random_check_viewrange(Point pos1, Point pos2) {
    double dist = euclidean_distance(pos1, pos2);
    double prob_in_viewrange = probability_in_viewrange(dist);
    std::uniform_real_distribution<double> unif(0, 1);
    return unif(mt) < prob_in_viewrange;
}

double Agent::probability_in_viewrange(double dist) {
    if (dist < D_MIN) {
        return DETECTION_PROBABILITY_IN_RANGE;
    } else if (dist >= D_MIN && dist < D_MAX) {
        return DETECTION_PROBABILITY_IN_RANGE * (D_MAX - dist) / (D_MAX - D_MIN);
    } else if (dist >= D_MAX) {
        return 0.0;
    }
    return 0.0;  // should never be reached, to prevent compiler warning
}