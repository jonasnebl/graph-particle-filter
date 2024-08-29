from utils import extend_perception
import numpy as np

class ConfidentTracker:
    def __init__(self, N_nodes, include_observations=True):
        self.include_observations = include_observations
        self.N_nodes = N_nodes
        self.probabilities = np.zeros((self.N_nodes,))
        self.constant_probabilities = np.zeros((self.N_nodes,))

    def add_observation(self, robot_perceptions):
        """Update P based on a set of robot perceptions.
        """
        if self.include_observations:
            robot_perceptions_extended = extend_perception(robot_perceptions)
            self.probabilities[np.unique(robot_perceptions_extended.flatten())] = 1

        return self.probabilities

    def predict(self):
        return np.zeros((self.N_nodes,))
