#include "simulation.h"

#include <algorithm>
#include <ctime>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>

#include "agent.h"
#include "warehouse_data.h"

Simulation::Simulation(double T_step, int N_humans, int N_robots)
    : _T_step(T_step), _N_humans(N_humans), _N_robots(N_robots) {
    // load graph
    nodes = warehouse_data::nodes;
    edges = warehouse_data::edges;
    edge_weights = warehouse_data::edge_weights;
    racks = warehouse_data::racks;
    node_polygons = warehouse_data::node_polygons;

    // random node index
    // generator
    std::random_device rd;
    mt = std::mt19937(rd());
    dist = std::uniform_int_distribution<int>(0, nodes.size() - 1);
    node_noise = std::normal_distribution<double>(0, 0.5);

    // initialize agents
    for (int i = 0; i < _N_humans; i++) {
        agents.push_back(Agent(T_step, true, this));
    }
    for (int i = 0; i < _N_robots; i++) {
        agents.push_back(Agent(T_step, false, this));
    }
}

double Simulation::get_node_noise() { return node_noise(mt); }

int Simulation::get_random_node_index() { return dist(mt); }

std::vector<std::vector<pybind11::dict>> Simulation::step(int N_steps) {
    auto result = std::vector<std::vector<pybind11::dict>>(N_steps);
    for (int i = 0; i < N_steps; i++) {
        // step all agents
        for (auto &agent : agents) {
            agent.step();
        }

        // log state for step
        std::vector<pybind11::dict> step_state;
        for (auto &agent : agents) {
            step_state.push_back(agent.log_state());
        }
        result[i] = step_state;
    }
    return result;
}

std::vector<int> Simulation::dijkstra(int start_node, int end_node) {
    // initialize arrays
    std::vector<double> distances(nodes.size(), std::numeric_limits<double>::infinity());
    distances[start_node] = 0;
    std::vector<int> predecessors(nodes.size(), std::numeric_limits<int>::max());
    std::vector<bool> considered(nodes.size(), false);

    // priority queue to
    // select the node with
    // the smallest distance
    using NodeDistPair = std::pair<double, int>;
    std::priority_queue<NodeDistPair, std::vector<NodeDistPair>, std::greater<NodeDistPair>> pq;
    pq.emplace(0, start_node);

    while (!pq.empty()) {
        auto [current_distance, current_node] = pq.top();
        pq.pop();

        if (considered[current_node]) continue;
        considered[current_node] = true;

        for (int j = 0; j < edges.size(); j++) {
            if (edges[j].first == current_node) {
                int neighbor = edges[j].second;
                double new_distance = current_distance + edge_weights[j];
                if (new_distance < distances[neighbor]) {
                    distances[neighbor] = new_distance;
                    predecessors[neighbor] = current_node;
                    pq.emplace(new_distance, neighbor);
                }
            }
        }
    }

    // extract and return
    // solution
    std::vector<int> optimal_path;
    for (int at = end_node; at != std::numeric_limits<int>::max(); at = predecessors[at]) {
        optimal_path.insert(optimal_path.begin(), at);
    }

    return optimal_path;
}
