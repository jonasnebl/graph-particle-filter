#ifndef SIMULATION_H
#define SIMULATION_H

#include <array>
#include <vector>
#include <iostream>
#include <queue>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>

#include "agent.h"

class Agent;

namespace py = pybind11;

class Simulation {
    public:
        Simulation(double T_step, int N_humans, int N_robots);
        // agent attributes
        std::size_t _N_humans;
        std::size_t _N_robots;
        const double _T_step;
        std::vector<Agent> agents;

        // warehouse structure
        std::vector<std::pair<double, double>> nodes;
        std::vector<std::pair<std::size_t, std::size_t>> edges;
        std::vector<double> edge_weights;
        std::vector<std::vector<std::pair<double, double>>> racks;
        std::vector<std::vector<std::pair<double, double>>> node_polygons;

        // utility functions 
        std::vector<std::vector<py::dict>> step(std::size_t N_steps);
        std::size_t get_random_node_index();
        double get_xy_noise();
        std::vector<std::size_t> dijkstra(std::size_t start, std::size_t end);   

    private:
        // utility function helper variables
        std::mt19937 mt;
        std::uniform_int_distribution<std::size_t> dist;
        std::normal_distribution<double> xy_noise;
};

#endif