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

#include "particle.h"

using Point = std::pair<double, double>;

class ParticleTracker {
   public:
    ParticleTracker(double T_step, int N_humans_max, int N_particles);
    std::vector<double> add_observation(std::vector<pybind11::dict> robot_observations);
    std::vector<double> predict();

    // warehouse graph
    std::vector<Point> nodes;
    std::vector<std::pair<int, int>> edges;
    std::vector<std::vector<int>> successor_edges;
    std::vector<double> edge_weights;
    std::vector<std::vector<Point>> racks;

   private:
    // prediction model
    Particle prediction_model(Particle particle);
    std::vector<std::vector<std::array<double, 3>>> pred_model_params;
    void save_pred_model_params() const;
    std::string pred_model_params_filename = "models/pred_model_params.json";

    // calculate node probabilities from internal sysstem state
    std::vector<double> calculate_edge_probabilities();

    // random number generator variables
    std::mt19937 mt;

    // particle filter attributes
    const double T_step;
    const int N_humans_max;
    const int N_particles;
    std::vector<Particle> particles;
};

#endif