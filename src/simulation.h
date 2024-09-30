#ifndef WAREHOUSESIM_SRC_SIMULATION_H
#define WAREHOUSESIM_SRC_SIMULATION_H

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

#include "agent.h"

class Agent;

using Point = std::pair<double, double>;

const double TRAJECTORY_XY_STDDEV = 0.5; // meters

class Simulation {
   public:
    Simulation(double T_step, int N_humans, int N_robots);
    // agent attributes
    int _N_humans;
    int _N_robots;
    double _T_step;
    std::vector<Agent> agents;

    // warehouse structure
    std::vector<Point> nodes;
    std::vector<std::pair<int, int>> edges;
    std::vector<double> edge_weights;
    std::vector<std::vector<Point>> racks;
    std::vector<int> staging_nodes;
    std::vector<int> storage_nodes;
    std::vector<int> exit_nodes;

    // utility functions
    std::vector<std::vector<pybind11::dict>> step(int N_steps);
    int get_random_node_index();
    double get_trajectory_xy_noise();
    std::vector<int> dijkstra(int start, int end);

   private:
    // utility function helper variables
    std::mt19937 mt;
    std::uniform_int_distribution<int> random_node_index;
    std::normal_distribution<double> trajectory_xy_noise;
};

#endif