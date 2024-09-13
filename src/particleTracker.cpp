#include "particleTracker.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <vector>

#include "agent.h"
#include "particle.h"
#include "warehouse_data.h"

ParticleTracker::ParticleTracker(double T_step, int N_humans_max, int N_particles)
    : T_step(T_step), N_humans_max(N_humans_max), N_particles(N_particles) {
    // load graph
    nodes = warehouse_data::nodes;
    edges = warehouse_data::edges;
    edge_weights = warehouse_data::edge_weights;
    racks = warehouse_data::racks;

    // calculate successor edges
    for (int i = 0; i < edges.size(); i++) {
        std::vector<int> successor_edges_of_i;
        for (int j = 0; j < edges.size(); j++) {
            if (edges[i].second == edges[j].first) {
                successor_edges_of_i.push_back(j);
            }
        }
        successor_edges.push_back(successor_edges_of_i);
    }

    // load prediction model parameters
    try {
        std::ifstream file(pred_model_params_filename);
        nlohmann::json json_data;
        file >> json_data;
        pred_model_params = json_data.get<std::vector<std::vector<std::array<double, 3>>>>();
    } catch (const std::exception& e) {
        // initialize new prediction model parameters
        for (int i = 0; i < edges.size(); i++) {
            std::vector<std::array<double, 3>> params_per_edge(
                successor_edges[i].size(),
                std::array<double, 3>({1 / static_cast<double>(successor_edges[i].size()), 5.0, 2.0}));
            pred_model_params.push_back(params_per_edge);
        }
        save_pred_model_params();
    }

    // initialize particles
    particles = std::vector<Particle>(
        N_particles, Particle(N_humans_max, &nodes, &edges, &racks, &successor_edges, &pred_model_params));

    // init random number generator
    std::random_device rd;
    mt = std::mt19937(rd());
}

std::vector<double> ParticleTracker::add_observation(std::vector<pybind11::dict> robot_observations) {
    std::vector<double> distances;
    for (int i = 0; i < N_particles; i++) {
        std::vector<double> robot_specific_distances;
        for (const auto& robot_perception : robot_observations) {
            // --- parse perceptions ---
            Point robot_position = robot_perception["ego_position"].cast<Point>();
            auto perceived_humans = robot_perception["perceived_humans"].cast<std::vector<pybind11::dict>>();

            // --- Simulate the measurement ---
            auto measurements = particles[i].simulate_measurement(robot_position);
;
            // --- Compute the distance metric ---
            double distance = 0.0;
            if (perceived_humans.size() != measurements.size()) {
                distance = 1000.0;
            }
            robot_specific_distances.push_back(distance);
        }

        // --- Combine distance metric for all robots ---
        distances.push_back(std::accumulate(robot_specific_distances.begin(), robot_specific_distances.end(), 0.0) /
                            robot_specific_distances.size());
    }

    // --- Remove the particles with the highest distance and fill up with particles with lower distance ---
    int N_keep = N_particles / 2;
    std::vector<int> range(N_particles);
    std::iota(range.begin(), range.end(), 0);
    std::sort(range.begin(), range.end(), [&distances](int i1, int i2) { return distances[i1] < distances[i2]; });
    for (int i = 0; i < N_keep; i++) {
        particles[i] = particles[range[i]];
    }
    for (int i = N_keep; i < N_particles; i++) {
        int random_particle = std::uniform_int_distribution<int>(0, N_keep - 1)(mt);
        particles[i] = Particle(particles[random_particle]);
    }
    return calculate_edge_probabilities();
}

std::vector<double> ParticleTracker::predict() {
    for (auto& particle : particles) {
        particle.predict(T_step);
    }
    return calculate_edge_probabilities();
}

std::vector<double> ParticleTracker::calculate_edge_probabilities() {
    std::vector<double> edge_probabilities(edges.size(), 0.0);
    for (int i = 0; i < edges.size(); i++) {
        for (const auto& particle : particles) {
            for (const auto& human : particle.humans) {
                if (human.edge == i) {
                    edge_probabilities[i] += 1 / N_particles;
                    break;
                }
            }
        }
    }
    return edge_probabilities;
}

void ParticleTracker::save_pred_model_params() const {
    nlohmann::json json_data = pred_model_params;
    std::ofstream file(pred_model_params_filename);
    file << json_data.dump(4);
}