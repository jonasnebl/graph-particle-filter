#ifndef WAREHOUSESIM_SRC_PARTICLEFILTER_H
#define WAREHOUSESIM_SRC_PARTICLEFILTER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "particle.h"
#include "warehouse_data.h"

const double T_WINDOW = 5 * 60;

using Point = std::pair<double, double>;

class ParticleTracker {
   public:
    ParticleTracker(double T_step, int N_tracks_init, int N_particles);
    std::pair<std::vector<pybind11::dict>, std::vector<Point>> merge_perceptions(
        std::vector<pybind11::dict> robot_perceptions);
    std::vector<double> add_merged_perceptions(std::vector<pybind11::dict> perceived_humans,
                                               std::vector<Point> robot_positions);
    int estimate_N_humans(int N_perceived);
    void add_one_track();
    void remove_one_track();

    std::vector<double> predict();

    graph_struct graph;

    // helper function for simulation
    static constexpr double HEADING_WEIGHT = 20.0;
    static std::tuple<double, double> edge_to_pose_distance_and_t(int edge, Point position,
                                                                  double heading,
                                                                  graph_struct &graph);

   private:
    // prediction model
    Particle prediction_model(Particle particle);

    // calc node probabilities from internal system state
    std::vector<double> calc_edge_probabilities();

    // helper functions
    Particle generate_new_particle_from_perception(Point perceived_pos, double position_stddev,
                                                   double perceived_heading, double heading_stddev);
    std::tuple<int, double> get_belonging_edge(Point position, double heading);
    std::vector<std::vector<int>> calc_assignment_cost_matrix(
        std::vector<pybind11::dict> perceived_humans, int particle_index);
    std::vector<int> assign_perceived_humans_to_internal_humans(
        std::vector<std::vector<int>> cost_matrix);
    void normalize_weights();
    void print_weights();
    static double heading_distance(double h1, double h2);
    double calc_effective_sample_size() const;

    // random number generator variables
    std::mt19937 mt;

    // particle filter attributes
    const double T_step;
    int N_tracks;  // not const because we can add and remove tracks
    const int N_particles;
    std::vector<std::vector<Particle>> particles;
    std::vector<double> particle_weights;
    std::deque<int> N_perceived_window;
};

#endif