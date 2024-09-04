#include "particleTracker.h"

#include "warehouse_data.h"

ParticleTracker::ParticleTracker(double T_step, int N_humans_max, int N_particles)
    : T_step(T_step), N_humans_max(N_humans_max), N_particles(N_particles) {

    T_history = 1; // seconds
    N_history = static_cast<int>(T_history / T_step);
    START_NODE = 1;

    human_positions.resize(N_history, std::vector<std::vector<uint8_t>>(N_particles, std::vector<uint8_t>(N_humans_max, START_NODE)));
    particle_weights.resize(N_particles, 1.0 / N_particles);
    RESAMPLE_THRESHOLD = 1e-4;

    // load graph
    nodes = warehouse_data::nodes;
    edges = warehouse_data::edges;
    edge_weights = warehouse_data::edge_weights;
    N_nodes = nodes.size();

    // init random number generator
    std::random_device rd;
    mt = std::mt19937(rd());    
}

std::vector<double> ParticleTracker::add_observation(std::vector<pybind11::dict> robot_observations) {
    for (const auto& robot_perception : robot_observations) {
        // --- parse perceptions ---
        auto observable_nodes = robot_perception["observable_nodes"].cast<std::vector<double>>();
        auto perceived_humans = robot_perception["perceived_humans"].cast<std::vector<pybind11::dict>>();

        std::vector<double> perceived_probabilities(observable_nodes.size(), 0.0);
        for (const auto& human : perceived_humans) {
            int belonging_node = human["belonging_node"].cast<int>();
            perceived_probabilities[belonging_node] = 1.0;
        }

        // --- sequential importance sampling ---
        std::vector<double> likelihood(N_particles, 0.0);
        for (int i = 0; i < N_particles; i++) {
            std::vector<double> expected_probabilities(observable_nodes.size(), 0.0);
            for (int j = 0; j < N_humans_max; j++) {
                int node_index = human_positions[0][i][j];
                if (observable_nodes[node_index] > 0.5) {
                    expected_probabilities[node_index] = 1.0;
                }
            }
            likelihood[i] = std::equal(expected_probabilities.begin(), expected_probabilities.end(), perceived_probabilities.begin()) ? 1.0 : 0.0;
        }

        for (int i = 0; i < N_particles; i++) {
            particle_weights[i] *= likelihood[i];
        }

        // --- Resampling ---
        std::vector<int> particles_to_be_resampled;
        std::vector<int> particles_to_be_kept;
        for (int i = 0; i < N_particles; i++) {
            if (particle_weights[i] < RESAMPLE_THRESHOLD) {
                particles_to_be_resampled.push_back(i);
            } else {
                particles_to_be_kept.push_back(i);
            }
        }
        if (particles_to_be_kept.size() >= 1) {
            std::uniform_int_distribution<> dis(0, particles_to_be_kept.size() - 1);
            for (int i = 0; i < particles_to_be_resampled.size(); i++) {
                    int random_particle = particles_to_be_kept[dis(mt)];
                    for(int j = 0; j < N_humans_max; j++) {
                        for(int k = 0; k < N_history; k++) {
                            human_positions[k][particles_to_be_resampled[i]][j] = human_positions[k][random_particle][j];
                        }
                    }
                    
            }
        }

        std::fill(particle_weights.begin(), particle_weights.end(), 1.0 / N_particles);
    }

    return calculate_node_probabilities();
}

std::vector<double> ParticleTracker::predict() {
    std::vector<std::vector<uint8_t>> new_human_positions;
    for (int i = 0; i < N_particles; i++) {
        for (int j = 0; j < N_humans_max; j++) {
            std::vector<int> history;
            for (int k = 0; k < N_history-1; k++) {
                human_positions[k+1][i][j] = human_positions[k][i][j];
                history.push_back(human_positions[k][i][j]);
            }
            human_positions[0][i][j] = prediction_model(history);
        }
    }
    return calculate_node_probabilities();
}

int ParticleTracker::prediction_model(std::vector<int> history) {
    int current_node = history[0];
    std::vector<int> adjacent_nodes;
    for (const auto& edge : edges) {
        if (edge.first == current_node) {
            adjacent_nodes.push_back(edge.second);
        }
    }

    std::vector<int> candidate_nodes = {current_node};
    candidate_nodes.insert(candidate_nodes.end(), adjacent_nodes.begin(), adjacent_nodes.end());

    std::vector<double> candidate_probabilities(candidate_nodes.size(), 0.0);
    candidate_probabilities[0] = 0.9;
    for (size_t i = 1; i < candidate_probabilities.size(); ++i) {
        candidate_probabilities[i] = 0.1 / (candidate_probabilities.size() - 1);
    }

    std::discrete_distribution<> dist(candidate_probabilities.begin(), candidate_probabilities.end());

    return candidate_nodes[dist(mt)];
}
    
std::vector<double> ParticleTracker::calculate_node_probabilities() {
    std::vector<double> node_probabilities(N_nodes, 0.0);
    for (int i = 0; i < N_nodes; ++i) {
        for (int j = 0; j < N_particles; ++j) {
            if (std::any_of(human_positions[0][j].begin(), human_positions[0][j].end(), [&i](int pos){ return pos == i; })) {
                node_probabilities[i] += particle_weights[j];
            }
        }
    }
    return node_probabilities;
}