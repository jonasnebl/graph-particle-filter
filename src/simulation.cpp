#include "simulation.h"
#include "warehouse_data.h"
#include <algorithm>
#include <limits>
#include <queue>
#include <numeric>
#include <iostream>
#include <ctime>
#include <random>

Simulation::Simulation(double T_step, int N_humans, int N_robots) 
: _T_step(T_step), _N_humans(N_humans), _N_robots(N_robots) {

    // load graph
    nodes = warehouse_data::nodes;
    edges = warehouse_data::edges;
    edge_weights = warehouse_data::edge_weights;
    racks = warehouse_data::racks;

    // random node index generator
    std::random_device rd;
    mt = std::mt19937(rd());
    dist = std::uniform_int_distribution<std::size_t>(0, nodes.size()-1);
    xy_noise = std::normal_distribution<double>(0, 0.05);

    // initialize agents
    for(int i = 0; i<_N_humans; i++) {
        agents.push_back(Agent(T_step, false, this));
    }
    for(int i = 0; i<_N_robots; i++) {
        agents.push_back(Agent(T_step, true, this));
    }    
}

double Simulation::get_xy_noise() {
    return xy_noise(mt);
}

std::size_t Simulation::get_random_node_index() {
    return dist(mt);
}

std::vector<std::vector<py::dict>> Simulation::step(std::size_t N_steps) {
    auto result = std::vector<std::vector<py::dict>>(N_steps);
    for(std::size_t i = 0; i<N_steps; i++) {

        // step all agents
        for (auto& agent : agents) {
            agent.step();
        }

        // log state for step
        std::vector<py::dict> step_state;
        for (auto& agent : agents) {
            step_state.push_back(agent.log_state());
        }
        result[i] = step_state;
    }
    return result;
}

std::vector<std::size_t> Simulation::dijkstra(std::size_t start_node, std::size_t end_node) {
    // initialize arrays
    std::vector<double> distances(nodes.size(), std::numeric_limits<double>::infinity());
    distances[start_node] = 0;
    std::vector<std::size_t> predecessors(nodes.size(), std::numeric_limits<std::size_t>::max());
    std::vector<bool> considered(nodes.size(), false);

    // priority queue to select the node with the smallest distance
    using NodeDistPair = std::pair<double, std::size_t>;
    std::priority_queue<NodeDistPair, std::vector<NodeDistPair>, std::greater<NodeDistPair>> pq;
    pq.emplace(0, start_node);

    while (!pq.empty()) {
        auto [current_distance, current_node] = pq.top();
        pq.pop();

        if (considered[current_node]) continue;
        considered[current_node] = true;

        for (std::size_t j = 0; j < edges.size(); j++) {
            if (edges[j].first == current_node) {
                std::size_t neighbor = edges[j].second;
                double new_distance = current_distance + edge_weights[j];
                if (new_distance < distances[neighbor]) {
                    distances[neighbor] = new_distance;
                    predecessors[neighbor] = current_node;
                    pq.emplace(new_distance, neighbor);
                }
            }
        }
    }

    // extract and return solution
    std::vector<std::size_t> optimal_path;
    if (distances[end_node] == std::numeric_limits<double>::infinity()) {
        std::cerr << "No path found from " << start_node << " to " << end_node << std::endl;
        return optimal_path; // no path found
    }

    for (std::size_t at = end_node; at != std::numeric_limits<std::size_t>::max(); at = predecessors[at]) {
        optimal_path.insert(optimal_path.begin(), at);
    }

    return optimal_path;
}
