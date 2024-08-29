from utils import extend_perception
import numpy as np

class AccurateTracker:
    def __init__(self, N_nodes, include_observations=True):
        self.include_observations = include_observations
        self.N_nodes = N_nodes
        self.probabilities = np.zeros((self.N_nodes,))
        self.mean_probabilities = np.zeros((self.N_nodes,))
        self.h = 0.001

    def add_observation(self, robot_perceptions):
        """Update P based on a set of robot perceptions.
        """

        robot_perceptions_extended = extend_perception(robot_perceptions)
        perceived_probabilities = np.zeros((self.N_nodes,))
        perceived_probabilities[np.unique(robot_perceptions_extended.flatten())] = 1
        self.mean_probabilities = self.h * perceived_probabilities + (1 - self.h) * self.mean_probabilities
        
        self.probabilities = self.mean_probabilities
        if self.include_observations:           
            self.probabilities[np.unique(robot_perceptions_extended.flatten())] = 1
        return self.probabilities

    def predict(self):
        return self.mean_probabilities
