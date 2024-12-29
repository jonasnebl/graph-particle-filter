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
#include "warehouse_data.h"

class Agent;

using Point = std::pair<double, double>;

const double PATH_XY_STDDEV = 0.4;  // meters

class Simulation {
   public:
    Simulation(double T_step, int N_humans, int N_robots, bool allow_warehouse_leaving);
    // agent attributes
    const int _N_humans;
    const int _N_robots;
    const double _T_step;
    const bool _allow_warehouse_leaving;
    std::vector<Agent> agents;

    // warehouse structure
    graph_struct graph;

    // utility functions
    std::vector<std::vector<pybind11::dict>> step(int N_steps);
    int get_random_node_index();
    double get_trajectory_xy_noise();
    static std::vector<int> dijkstra(int start, int end, const graph_struct &graph);

   private:
    // utility function helper variables
    std::mt19937 mt;
    std::uniform_int_distribution<int> random_node_index;
    std::normal_distribution<double> trajectory_xy_noise;
};

#endif  // WAREHOUSESIM_SRC_SIMULATION_H