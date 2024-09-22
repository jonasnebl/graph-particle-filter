from .tracker import Tracker
import numpy as np


class ConfidentTracker(Tracker):
    def __init__(self, N_robots, include_observations=True):
        super().__init__(N_robots)

        self.include_observations = include_observations
        self.probabilities = np.zeros((self.N_edges,))
        self.constant_probabilities = np.zeros((self.N_edges,))

    def add_observation(self, robot_perceptions):
        """Update P based on a set of robot perceptions."""
        if self.include_observations:
            perceived_human_node_belongings = self.get_perceived_human_node_belongings(robot_perceptions)
            self.observed_probabilities = np.zeros((self.N_edges,))
            try:
                self.observed_probabilities[np.unique(perceived_human_node_belongings.flatten())] = 1
            except:
                print(self.observed_probabilities)
            self.probabilities = self.observed_probabilities

        return self.probabilities

    def predict(self):
        return np.zeros((self.N_edges,))
