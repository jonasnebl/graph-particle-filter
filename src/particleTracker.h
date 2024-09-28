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
    std::vector<double> add_observation(std::vector<pybind11::dict> robot_perceptions);
    std::vector<double> predict();
    void save_pred_model_params() const;

    // warehouse graph
    std::vector<Point> nodes;
    std::vector<std::pair<int, int>> edges;
    std::vector<std::vector<int>> successor_edges;
    std::vector<double> edge_weights;
    std::vector<std::vector<Point>> racks;

    // helper function for simulation
    static double distance_of_point_to_edge(Point point, Point edge_start, Point edge_end);

   private:
    // prediction model
    Particle prediction_model(Particle particle);
    std::vector<std::vector<std::array<double, 3>>> pred_model_params;
    std::string pred_model_params_filename = "models/pred_model_params.json";

    // calculate node probabilities from internal sysstem state
    std::vector<double> calculate_edge_probabilities();
    std::vector<double> calculate_edge_probabilities_one_human(int index_human);
    std::vector<double> weighted_calculate_edge_probabilities_one_human(int index_human);
    std::vector<double> weighted_calculate_edge_probabilities();
    void normalize_weights(int index_human);
    void print_weights(int index_human);

    // helper functions
    void add_single_observation(pybind11::dict robot_perception);
    int get_belonging_edge(Point position, double heading);
    std::vector<int> assign_perceived_humans_to_internal_humans(Point robot_position, std::vector<pybind11::dict> perceived_humans);
    void update_particles(int i, std::vector<double> distances, Point perceived_pos, double perceived_heading);
    std::pair<int, int> find_max_element_index(const std::vector<std::vector<double>>& matrix);

    // random number generator variables
    std::mt19937 mt;

    // particle filter attributes
    const double T_step;
    const int N_humans_max;
    const int N_particles;
    std::vector<std::vector<Particle>> particles;
    std::vector<std::vector<double>> particle_weights;
};

#endif