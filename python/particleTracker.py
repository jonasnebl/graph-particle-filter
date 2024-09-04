import numpy as np
from .utils import load_warehouse_data_from_json


class ParticleTracker:
    def __init__(self, T_step, N_humans_max=3, N_particles=100):
        """ParticleTracker using a particle filter to track humans"""
        self.N_humans_max = N_humans_max
        self.N_particles = N_particles
        self.T_history = 1  # 10 seconds
        self.T_step = T_step
        self.N_history = int(self.T_history / self.T_step)
        START_NODE = 1
        self.human_positions = START_NODE * np.ones(
            (self.N_history, self.N_particles, self.N_humans_max), dtype=np.uint8
        )
        self.particle_weights = 1 / self.N_particles * np.ones((self.N_particles,))
        self.RESAMPLE_THRESHOLD = 1e-4

        self.nodes, self.edges, self.edge_weights, self.polygons = load_warehouse_data_from_json()
        self.N_nodes = len(self.nodes)

    def add_observation(self, robot_perceptions):
        """Update P based on a set of robot perceptions."""

        for robot_perception in robot_perceptions:
            # --- parse perceptions ---
            robot_position = robot_perception["ego_position"]
            observable_nodes = np.array(robot_perception["observable_nodes"])
            perceived_humans = robot_perception["perceived_humans"]

            perceived_probabilities = np.zeros((observable_nodes,))
            for human in perceived_humans:
                perceived_probabilities[human["belonging_node"]] = 1

            # --- sequential importance sampling ---
            # TODO more sophisticated likelihood
            likelihood = np.zeros((self.N_particles,))
            for i in range(self.N_particles):
                expected_probabilities = np.zeros(observable_nodes.shape)
                for j in range(self.N_humans_max):
                    node_index = self.human_positions[0, i, j]
                    if observable_nodes[node_index] > 0.5:
                        expected_probabilities[node_index] = 1
                try:
                    likelihood[i] = float(np.allclose(expected_probabilities, perceived_probabilities))
                except:
                    likelihood[i] = 0

            self.particle_weights *= likelihood

            # --- Resampling ---
            resampling_mask = self.particle_weights < self.RESAMPLE_THRESHOLD
            if np.sum(~resampling_mask) > 0:
                for i in range(self.N_particles):
                    if resampling_mask[i]:
                        # copy one random particle
                        random_node = np.random.choice(a=np.arange(self.N_particles)[~resampling_mask])
                        self.human_positions[:, i, :] = self.human_positions[:, random_node, :]

            self.particle_weights = 1 / self.N_particles * np.ones((self.N_particles,))

        return self._calculate_node_probabilities()

    def predict(self):
        """Predict all particles by one timestep T_step"""
        new_human_positions = np.zeros((self.N_particles, self.N_humans_max), dtype=np.uint8)
        for i in range(self.N_particles):
            for j in range(self.N_humans_max):
                new_human_positions[i, j] = self.prediction_model(self.human_positions[:, i, j])
        self.human_positions = np.concatenate(([new_human_positions], self.human_positions[:-1]), axis=0)
        return self._calculate_node_probabilities()

    def prediction_model(self, history):
        """Predict next node based on node history"""
        current_node = history[0]
        adjacent_nodes = [edge[1] for edge in self.edges if edge[0] == current_node]
        candidate_nodes = np.array([current_node] + adjacent_nodes)
        candidate_probabilities = np.zeros(candidate_nodes.shape)
        candidate_probabilities[0] = 0.9
        candidate_probabilities[1:] = 0.1 / (candidate_probabilities.size - 1)

        return np.random.choice(candidate_nodes, p=candidate_probabilities)

    def _calculate_node_probabilities(self):
        """Calculate node probabilities based on current particles state"""
        node_probabilites = np.zeros((self.N_nodes,))
        for i in range(self.N_nodes):
            node_probabilites[i] = np.dot(self.particle_weights, np.any(self.human_positions[0] == i, axis=1))
        return node_probabilites
