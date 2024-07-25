#include "simulation.h"
#include "human.h"
#include "robot.h"
#include "graph_data.h"
#include <algorithm>
#include <limits>
#include <numeric>

Simulation::Simulation(double T_step, int N_humans, int N_robots) 
: _T_step(T_step), _N_humans(N_humans), _N_robots(N_robots) {
    // load graph
    nodes = graph_data::nodes;
    edges = graph_data::edges;
    edge_weights = graph_data::edge_weights;

    // initialize humans
    agents = std::vector<Agent*>(N_humans + N_robots);
    for(std::size_t i = 0; i<N_humans; i++) {
        agents[i] = new Human(_T_step, this);
    }
    for(std::size_t i = N_humans; i<N_humans + N_robots; i++) {
        agents[i] = new Robot(_T_step, this);
    }
}

std::vector<std::vector<py::dict>> Simulation::step(std::size_t N_steps) {
    auto result = std::vector<std::vector<py::dict>>(N_steps);
    for(std::size_t i = 0; i<N_steps; i++) {

        // step all agents
        for (auto& agent : agents) {
            agent->step();
        }

        // log state for step
        std::vector<py::dict> step_state;
        for (auto& agent : agents) {
            step_state.push_back(agent->log_state());
        }
        result[i] = step_state;
    }
    return result;
}

std::vector<std::size_t> Simulation::dijkstra(std::size_t start_node, std::size_t end_node) {
    // initialize arrays
    std::vector<double> distances(nodes.size());
    std::fill(distances.begin(), distances.end(), std::numeric_limits<double>::infinity());
    distances[start_node] = 0;
    std::vector<std::size_t> predecessors(nodes.size());
    std::vector<bool> considered(nodes.size());
    std::fill(considered.begin(), considered.end(), false);

    // perform the algorithm
    std::size_t current_node = start_node;
    for(std::size_t i = 0; i<nodes.size(); i++) {
        for(std::size_t j = 0; j<edges.size(); j++) {
            if(edges[j].first == current_node) {
                if(edge_weights[j] + distances[current_node] < distances[edges[j].second]) {
                    predecessors[edges[j].second] = current_node;
                    distances[edges[j].second] = edge_weights[j] + distances[current_node];
                }
            }
        }
        // find next node
        considered[current_node] = true;
        std::vector<std::size_t> sorted_indices(nodes.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(), 
            [&](std::size_t i, std::size_t j){
                return distances[i] < distances[j];
            });
        std::size_t k = 0;
        auto next_node_candidate = sorted_indices[k];
        while(considered[next_node_candidate]) {
            next_node_candidate = sorted_indices[++k];
        };
        current_node = next_node_candidate;
    }

    // extract and return solution
    std::vector<std::size_t> optimal_path;
    current_node = end_node;
    while(current_node != start_node) {
        optimal_path.insert(optimal_path.begin(), current_node);
        current_node = predecessors[current_node];
    }
    optimal_path.insert(optimal_path.begin(), start_node);
    return optimal_path;
}
