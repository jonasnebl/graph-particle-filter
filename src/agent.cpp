#include "agent.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include "simulation.h"
#include "particleTracker.h"

Agent::Agent(double T_step, bool is_human, Simulation *simulation)
    : _T_step(T_step), _is_human(is_human), _simulation(simulation) {
    int start_node_index = _is_human ? DROPOFF_HUMANS : DROPOFF_ROBOTS;
    position = (_simulation->nodes)[start_node_index];
    path = std::deque<Point>();
    add_node_to_deque(start_node_index);

    std::random_device rd;
    mt = std::mt19937(rd());
    position_noise = std::normal_distribution<double>(0, 0.0);
    heading_noise = std::normal_distribution<double>(0, 0.0);
    velocity_noise = std::normal_distribution<double>(0, 0.0);
}

void Agent::step() {
    while (path.size() <= 10) {
        add_new_job_to_deque();
    }

    double dist_remaining = velocity * _T_step;
    while (dist_remaining > 0) {
        double dist_x = path.front().first - position.first;
        double dist_y = path.front().second - position.second;
        heading = std::atan2(dist_y, dist_x);
        double dist = distance(position, path.front());
        if (dist < dist_remaining) {
            path.pop_front();
        } else {
            position.first += dist_remaining * dist_x / dist;
            position.second += dist_remaining * dist_y / dist;
        }
        dist_remaining -= dist;
    }
}

double Agent::distance(Point p1, Point p2) {
    return std::sqrt(std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2));
}

pybind11::dict Agent::log_state() {
    pybind11::dict state;
    state["ego_position"] = position;
    if (_is_human) {
        state["type"] = "human";
        state["belonging_edge"] = get_belonging_edge();
    } else {
        state["type"] = "robot";
        auto perceived_humans = perceive_humans();
        state["perceived_humans"] = perceived_humans;
    }
    return state;
}

int Agent::get_belonging_edge() {
    std::vector<double> distances;
    for (int i = 0; i < (_simulation->edges).size(); i++) {
        Point p1 = (_simulation->nodes)[(_simulation->edges)[i].first];
        Point p2 = (_simulation->nodes)[(_simulation->edges)[i].second];
        double cartesian_distance_to_edge = ParticleTracker::distance_of_point_to_edge(position, p1, p2);
        double heading_difference =
            std::abs(heading - std::atan2(p2.second - p1.second, p2.first - p1.first));
        distances.push_back(cartesian_distance_to_edge + 2 * heading_difference);
    }
    return std::min_element(distances.begin(), distances.end()) - distances.begin();
}

void Agent::add_new_job_to_deque() {
    // every job consists of a path to a random node, a break at that node,
    // and a path back to the start node, and a break at the start node the
    // break nodes will be denoted by the index occuring twice in the path queue
    int DROP_INDEX = _is_human ? DROPOFF_HUMANS : DROPOFF_ROBOTS;

    int end_node = DROP_INDEX;
    while (end_node == DROP_INDEX) {
        end_node = _simulation->get_random_node_index();
    }

    std::vector<int> path_to_target = _simulation->dijkstra(DROP_INDEX, end_node);
    for (int i = 1; i < path_to_target.size(); i++) {  // exclude first element
        add_node_to_deque(path_to_target[i]);
    }
    std::reverse(path_to_target.begin(), path_to_target.end());
    for (int i = 1; i < path_to_target.size(); i++) {  // exclude first element
        add_node_to_deque(path_to_target[i]);
    }
}

void Agent::add_node_to_deque(int node_index) {
    double node_x = (_simulation->nodes)[node_index].first + _simulation->get_node_noise();
    double node_y = (_simulation->nodes)[node_index].second + _simulation->get_node_noise();
    path.push_back({node_x, node_y});
}

std::vector<pybind11::dict> Agent::perceive_humans() {
    std::vector<pybind11::dict> result;
    for (int i = 0; i < _simulation->_N_humans + _simulation->_N_robots; i++) {
        if ((_simulation->agents)[i]._is_human) {
            Agent human = _simulation->agents[i];
            if (check_viewline(position, human.position, _simulation->racks)) {
                double dist = distance(position, human.position);
                pybind11::dict perceived_human;
                auto noisy_position = human.position;
                noisy_position.first += position_noise(mt) * dist;
                noisy_position.second += position_noise(mt) * dist;
                perceived_human["position"] = noisy_position;
                perceived_human["heading"] = human.heading + heading_noise(mt) * dist;
                perceived_human["velocity"] = human.velocity + velocity_noise(mt) * dist;
                perceived_human["belonging_edge"] = human.get_belonging_edge();
                result.push_back(perceived_human);
            }
        }
    }
    return result;
}

bool Agent::check_viewline(Point pos1, Point pos2, std::vector<std::vector<Point>> racks) {
    for (const auto &polygon : racks) {
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

// Function to check if two
// line segments intersect
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