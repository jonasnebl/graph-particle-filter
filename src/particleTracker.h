#ifndef WAREHOUSESIM_SRC_PARTICLEFILTER_H
#define WAREHOUSESIM_SRC_PARTICLEFILTER_H

#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using Point = std::pair<double, double>;

class ParticleTracker {
   public:
    ParticleTracker(double T_step, int N_humans_max, int N_particles);
    std::vector<double> add_observation(std::vector<pybind11::dict> robot_observations);
    std::vector<double> predict();

   private:
    int prediction_model(std::vector<int> history);
    std::vector<double> calculate_node_probabilities();

    // random number generator variables
    std::mt19937 mt;
    std::uniform_int_distribution<int> dist;

    const double T_step;
    const int N_humans_max;
    const int N_particles;

    double T_history;
    
    int N_history;
    int START_NODE;
    std::vector<std::vector<std::vector<uint8_t>>> human_positions;
    std::vector<double> particle_weights;
    double RESAMPLE_THRESHOLD;

    std::vector<Point> nodes;
    std::vector<std::pair<int, int>> edges;
    std::vector<double> edge_weights;
    int N_nodes;
};


#endif