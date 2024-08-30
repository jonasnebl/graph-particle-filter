from .tracker import Tracker
import numpy as np
import os
from .constants import *

class AccurateTracker(Tracker):
    def __init__(self, N_robots, include_observations=True, train=False):
        super().__init__(N_robots)

        self.include_observations = include_observations
        self.train = train
        self.probabilities = np.zeros((self.N_nodes,))

        self.mean_probabilities_filename = os.path.join(
            MODEL_PATH, 'accurateTracker_mean_probabilities.txt')
        if self.train:
            self.mean_probabilities = np.zeros((self.N_nodes,))
            self.observed_probabilities_list = []
        else:
            self.mean_probabilities = np.loadtxt(self.mean_probabilities_filename)

    def add_observation(self, robot_perceptions):
        """Update P based on a set of robot perceptions.
        """
        if self.train or self.include_observations:
            perceived_human_node_belongings = self.get_perceived_human_node_belongings(robot_perceptions)
            self.observed_probabilities = np.zeros((self.N_nodes,))
            try:
                self.observed_probabilities[np.unique(perceived_human_node_belongings.flatten())] = 1
            except:
                print(self.observed_probabilities)
            if self.train:
                self.observed_probabilities_list.append(self.observed_probabilities)

        self.probabilities = self.mean_probabilities.copy()
        if self.include_observations:
            self.probabilities[self.observed_probabilities > 0.5] = 1

        return self.probabilities
    
    def save_trained_model(self):
        if not self.train:
            raise ValueError('Model needs to be in training mode to save it.')
        else:
            np.savetxt(self.mean_probabilities_filename, 
                       np.array(self.observed_probabilities_list).mean(axis=0))

    def predict(self):
        return self.mean_probabilities
