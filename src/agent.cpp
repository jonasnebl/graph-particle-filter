#include "agent.h"
#include "simulation.h"

Agent::Agent(double T_step, Simulation* simulation) {
    _simulation = simulation;
    _T_step = T_step;
    std::size_t random_node_index = std::rand() % (_simulation->nodes).size();
    pose = {(_simulation->nodes)[random_node_index].first, 
            (_simulation->nodes)[random_node_index].second, 
            0};
    add_new_job_to_queue(random_node_index);
}

void Agent::step() {
    if(path.empty()) {
        std::size_t random_node_index = std::rand() % (_simulation->nodes).size();
        add_new_job_to_queue(random_node_index);
    }
    std::size_t next_node = path.front();
    path.pop();
    pose = {(_simulation->nodes)[next_node].first, 
            (_simulation->nodes)[next_node].second, 
            0};
}

void Agent::add_new_job_to_queue(std::size_t start_node) {
    std::size_t end_node = std::rand() % (_simulation->nodes).size();
    auto path = _simulation->dijkstra(start_node, end_node);
    for(auto node : path) {
        this->path.push(node);
    }
}