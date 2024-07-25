#ifndef SIMULATION_H
#define SIMULATION_H

#include <array>
#include <vector>
#include <iostream>
#include <queue>
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>

#include "agent.h"

class Agent;

namespace py = pybind11;

class Simulation {
    public:
        Simulation(double T_step, int N_humans, int N_robots);
        std::vector<std::vector<py::dict>> step(std::size_t N_steps);

        // warehouse structure
        std::vector<std::pair<double, double>> nodes;
        std::vector<std::pair<std::size_t, std::size_t>> edges;
        std::vector<double> edge_weights;

        // utility functions 
        std::vector<std::size_t> dijkstra(std::size_t start, std::size_t end);   
               
    private:
        // agent attributes
        const double _T_step;
        std::size_t _N_humans;
        std::size_t _N_robots;
        std::vector<Agent*> agents;
};

#endif