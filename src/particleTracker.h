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

const double EDGE_CHANGE_THRESHOLD = 0.9;
const double N_SIGMA = 4.0;
const double HEADING_WEIGHT = 0.5;
const double REL_RESAMPLE_THRESHOLD = 0.01;

using Point = std::pair<double, double>;

class ParticleTracker {
   public:
    ParticleTracker(double T_step, int N_tracks_init, int N_particles, int window_length);
    std::pair<std::vector<pybind11::dict>, std::vector<Point>> merge_perceptions(
        std::vector<pybind11::dict> robot_perceptions);
    std::vector<double> add_merged_perceptions(std::vector<pybind11::dict> perceived_humans,
                                               std::vector<Point> robot_positions);
    int estimate_N_humans(int N_perceived);
    void add_one_track();
    void remove_one_track();
    std::pair<std::vector<std::pair<int, int>>, std::vector<std::pair<int, double>>>
    calc_training_data();

    std::vector<double> predict();

    graph_struct graph;

    // helper function for simulation
    static double heading_distance(double h1, double h2);

   private:
    // prediction model
    Particle prediction_model(Particle particle);

    // calc node probabilities from internal system state
    std::vector<double> calc_edge_probabilities();
    std::vector<std::vector<double>> calc_individual_edge_probabilities();

    // helper functions
    std::vector<int> assign_perceived_humans_to_internal_humans(
        std::vector<pybind11::dict> perceived_humans, int particle_index);
    void normalize_weights();
    void print_weights();
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

    // training data generation
    std::vector<double> previous_edge_probabilities = std::vector<double>(graph.edges.size(), 0.0);
    std::vector<bool> edge_since_last_change_over_threshold =
        std::vector<bool>(graph.edges.size(), false);
    std::vector<double> edge_time_since_last_change = std::vector<double>(graph.edges.size(), 0.0);
};

#endif