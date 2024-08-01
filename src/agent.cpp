#include "agent.h"
#include "simulation.h"
#include <iostream>
#include <algorithm>
#include <cmath>

Agent::Agent(double T_step, bool is_human, Simulation* simulation) 
    : _T_step(T_step), _is_human(is_human), _simulation(simulation) {
    auto start_node_index = _is_human ? DROPOFF_HUMANS : DROPOFF_ROBOTS;
    pose = {(_simulation->nodes)[start_node_index].first, 
            (_simulation->nodes)[start_node_index].second, 
            0};
    path = std::deque<std::pair<double, double>>();
    add_node_to_deque(start_node_index);
}

void Agent::step() {

    while(path.size() <= 10) {
        add_new_job_to_deque();
    }

    double dist_remaining = speed * _T_step;
    while(dist_remaining > 0) {
        double dist_x = path.front().first - pose[0];
        double dist_y = path.front().second - pose[1];
        double dist = std::sqrt(dist_x*dist_x + dist_y*dist_y);
        if(dist < dist_remaining) {
            path.pop_front(); 
        } else {
            pose[0] += dist_remaining * dist_x / dist;
            pose[1] += dist_remaining * dist_y / dist;   
            pose[2] = std::atan2(dist_y, dist_x);    
        }
        dist_remaining -= dist;
    }
}

py::dict Agent::log_state() {
    py::dict state;
    state["x"] = pose[0];
    state["y"] = pose[1];
    state["theta"] = pose[2];

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
    // the break nodes will be denoted by the index occuring twice in the path queue
    std::size_t DROP_INDEX = _is_human ? DROPOFF_HUMANS : DROPOFF_ROBOTS;

    auto end_node = DROP_INDEX;
    while(end_node == DROP_INDEX) {
        end_node = _simulation->get_random_node_index();
    }

    auto path_to_target = _simulation->dijkstra(DROP_INDEX, end_node);
    for(std::size_t i = 1; i<path_to_target.size(); i++) { // exclude first element
        add_node_to_deque(path_to_target[i]);
    }
    std::reverse(path_to_target.begin(), path_to_target.end());
    for(std::size_t i = 1; i<path_to_target.size(); i++) { // exclude first element
        add_node_to_deque(path_to_target[i]);
    }
}

void Agent::add_node_to_deque(std::size_t node_index) {
    auto node_x = (_simulation->nodes)[node_index].first + _simulation->get_xy_noise();
    auto node_y = (_simulation->nodes)[node_index].second + _simulation->get_xy_noise();
    path.push_back({node_x, node_y});
}

std::vector<std::pair<double, double>> Agent::perceive_humans() {
    std::vector<std::pair<double, double>> result;
    for(std::size_t i = 0; i< _simulation->_N_humans + _simulation->_N_robots; i++) {
        if ((_simulation->agents)[i]._is_human) {
            auto pose_human = (_simulation->agents[i]).pose;
            std::pair<double, double> robot_pos = {pose[0], pose[1]};
            std::pair<double, double> human_pos = {pose_human[0], pose_human[1]};
            bool observable = true;

            for (const auto& polygon : _simulation->racks) {
                for (size_t i = 0; i < polygon.size(); ++i) {
                    std::pair<double, double> p1 = polygon[i];
                    std::pair<double, double> p2 = polygon[(i + 1) % polygon.size()];

                    if (do_intersect(robot_pos, human_pos, p1, p2)) {
                        observable = false; // Obstruction found
                        break;
                    }
                }
            }
            if (observable) {
                result.push_back({pose_human[0], pose_human[1]});
            }
        }
    }
    return result;
}

std::vector<std::pair<double, double>> Agent::extend_perception(std::vector<std::pair<double, double>> perceived_humans) {
    std::vector<std::pair<double, double>> result((_simulation->nodes).size());
    std::fill(result.begin(), result.end(), std::make_pair<double, double>(0.0, 0.0));

    for (const auto& mean_pos_human : perceived_humans) {
        std::vector<double> distances((_simulation->nodes).size());
        for(std::size_t i = 0; i<(_simulation->nodes).size(); i++) {
            distances[i] = std::sqrt(std::pow(mean_pos_human.first - (_simulation->nodes)[i].first, 2) + 
                                     std::pow(mean_pos_human.second - (_simulation->nodes)[i].second, 2));
        }
        auto min_index = std::min_element(distances.begin(), distances.end()) - distances.begin();
        result[min_index].first = 1;
    }
    return result;
}

// Function to check if two line segments intersect
bool Agent::do_intersect(std::pair<double, double> p1, 
                         std::pair<double, double> q1, 
                         std::pair<double, double> p2, 
                         std::pair<double, double> q2) {
    auto orientation = [](std::pair<double, double> p, std::pair<double, double> q, std::pair<double, double> r) {
        double val = (q.second - p.second) * (r.first - q.first) - (q.first - p.first) * (r.second - q.second);
        if (val == 0) return 0;  // collinear
        return (val > 0) ? 1 : 2; // clock or counterclock wise
    };

    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    if (o1 != o2 && o3 != o4) return true;

    return false;
}