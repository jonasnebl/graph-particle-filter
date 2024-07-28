#include "agent.h"
#include "simulation.h"
#include <iostream>

Agent::Agent(double T_step, bool is_human, Simulation* simulation) 
    : _T_step(T_step), _is_human(is_human), _simulation(simulation) {
    auto start_node_index = _is_human ? DROPOFF_HUMANS : DROPOFF_ROBOTS;
    pose = {(_simulation->nodes)[start_node_index].first, 
            (_simulation->nodes)[start_node_index].second, 
            0};
    path = std::deque<std::size_t>();
    path.push_front(start_node_index);
}

void Agent::step() {

    while(path.size() <= 10) {
        add_new_job_to_deque();
    }

    double dist_remaining = speed * _T_step;
    while(dist_remaining > 0) {
        auto current_destination_index = path.front();
        double dist_x = (_simulation->nodes)[current_destination_index].first - pose[0];
        double dist_y = (_simulation->nodes)[current_destination_index].second - pose[1];
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
        state["perception"] = perceive_humans();
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
        path.push_back(path_to_target[i]);
    }
    std::reverse(path_to_target.begin(), path_to_target.end());
    for(std::size_t i = 1; i<path_to_target.size(); i++) { // exclude first element
        path.push_back(path_to_target[i]);
    }
}

std::vector<std::pair<double, double>> Agent::perceive_humans() {
    std::vector<std::pair<double, double>> result;
    for(std::size_t i = 0; i< _simulation->_N_humans + _simulation->_N_robots; i++) {
        if ((_simulation->agents)[i]._is_human) {
            auto pose_human = (_simulation->agents[i]).pose;
            // calculate sight connection
            // if (sight_connection(pose_human, pose)) {
            if (true) {
                result.push_back({pose_human[0], pose_human[1]});
            }
        }
    }
    return result;
}