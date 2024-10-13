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
#include "warehouse_data.h"

using Point = std::pair<double, double>;

class ParticleTracker {
   public:
    ParticleTracker(double T_step, int N_humans_max, int N_particles);
    std::vector<std::vector<double>> add_observation(std::vector<pybind11::dict> robot_perceptions);
    std::vector<std::vector<double>> predict();

    graph_struct graph;

    // helper function for simulation
    static std::tuple<double, double> distance_of_point_to_edge(Point point, Point edge_start, Point edge_end);
    static double heading_distance(double heading1, double heading2);
    static constexpr double HEADING_WEIGHT = 20.0;
    static int find_edge_relative_index(int edge, int next_edge, graph_struct &graph);

   private:
    // prediction model
    Particle prediction_model(Particle particle);
    std::vector<std::vector<std::array<double, 3>>> pred_model_params;
    std::string pred_model_params_filename = "models/pred_model_params.json";
    
    std::vector<std::vector<double>> calc_prob_distance_matrix();

    // calc node probabilities from internal sysstem state
    std::vector<double> calc_edge_probabilities_one_human(int index_human);
    std::vector<std::vector<double>> calc_individual_edge_probabilities();

    // helper functions
    std::pair<std::vector<Point>, std::vector<pybind11::dict>> merge_perceptions(std::vector<pybind11::dict> robot_perceptions);
    Particle generate_new_particle_from_perception(Point perceived_pos, double position_stddev, double perceived_heading, double heading_stddev);
    std::tuple<int, double> get_belonging_edge(Point position, double heading);
    std::vector<std::function<int()>> assign_perceived_humans_to_internal_humans(std::vector<pybind11::dict> perceived_humans);
    void normalize_weights(int index_human);
    void print_weights(int index_human);

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