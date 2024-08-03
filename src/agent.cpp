#include "agent.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "simulation.h"

Agent::Agent(double T_step, bool is_human, Simulation *simulation)
    : _T_step(T_step), _is_human(is_human), _simulation(simulation) {
    auto start_node_index = _is_human ? DROPOFF_HUMANS : DROPOFF_ROBOTS;
    position = (_simulation->nodes)[start_node_index];
    path = std::deque<Point>();
    add_node_to_deque(start_node_index);
}

void Agent::step() {
    while (path.size() <= 10) {
        add_new_job_to_deque();
    }

    double dist_remaining = speed * _T_step;
    while (dist_remaining > 0) {
        double dist_x = path.front().first - position.first;
        double dist_y = path.front().second - position.second;
        double dist = std::sqrt(dist_x * dist_x + dist_y * dist_y);
        if (dist < dist_remaining) {
            path.pop_front();
        } else {
            position.first += dist_remaining * dist_x / dist;
            position.second += dist_remaining * dist_y / dist;
        }
        dist_remaining -= dist;
    }
}

pybind11::dict Agent::log_state() {
    pybind11::dict state;

    state["x"] = position.first;
    state["y"] = position.second;

    if (_is_human) {
        state["type"] = "human";
    } else {
        state["type"] = "robot";

        auto perceived_humans = perceive_humans();
        state["perception"] = perceived_humans;
        state["perception_extended"] = extend_perception(perceived_humans);
    }

    return state;
}

void Agent::add_new_job_to_deque() {
    // every job consists of a path to a random node, a break at that node,
    // and a path back to the start node, and a break at the start node
    // the break nodes will be denoted by the index occuring twice in the path
    // queue
    int DROP_INDEX = _is_human ? DROPOFF_HUMANS : DROPOFF_ROBOTS;

    int end_node = DROP_INDEX;
    while (end_node == DROP_INDEX) {
        end_node = _simulation->get_random_node_index();
    }

    std::vector<int> path_to_target =
        _simulation->dijkstra(DROP_INDEX, end_node);
    for (int i = 1; i < path_to_target.size(); i++) {  // exclude first element
        add_node_to_deque(path_to_target[i]);
    }
    std::reverse(path_to_target.begin(), path_to_target.end());
    for (int i = 1; i < path_to_target.size(); i++) {  // exclude first element
        add_node_to_deque(path_to_target[i]);
    }
}

void Agent::add_node_to_deque(int node_index) {
    double node_x =
        (_simulation->nodes)[node_index].first + _simulation->get_xy_noise();
    double node_y =
        (_simulation->nodes)[node_index].second + _simulation->get_xy_noise();
    path.push_back({node_x, node_y});
}

std::vector<Point> Agent::perceive_humans() {
    std::vector<Point> result;
    for (int i = 0; i < _simulation->_N_humans + _simulation->_N_robots; i++) {
        if ((_simulation->agents)[i]._is_human) {
            Point position_human = (_simulation->agents[i]).position;
            if (check_viewline(position, position_human)) {
                result.push_back(position_human);
            }
        }
    }
    return result;
}

bool Agent::check_viewline(Point pos1, Point pos2) {
    for (const auto &polygon : _simulation->racks) {
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

std::vector<Point> Agent::extend_perception(
    std::vector<Point> perceived_humans) {
    std::vector<Point> result((_simulation->nodes).size());
    std::fill(result.begin(), result.end(),
              std::make_pair<double, double>(0.0, 0.0));

    for (int i = 0; i < (_simulation->nodes).size(); i++) {
        // 1. Calculate confidence for each node by evaluating visibility
        result[i].second = static_cast<double>(
            check_viewline(position, (_simulation->nodes)[i]));

        // 2. Calculate probability of human presence at each node for each
        // human
        for (const auto &mean_pos_human : perceived_humans) {
            if (is_point_in_polygon(mean_pos_human,
                                    (_simulation->node_polygons)[i])) {
                result[i].first = 1.0;
            }
        }
    }
    return result;
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