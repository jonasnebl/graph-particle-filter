"""
Python wrapper for the C++ ParticleTracker class.
Enables documentation and type hints for the ParticleTracker class.
"""

import numpy as np
import sys

sys.path.append("build/")  # allos to import cpp_utils
from cpp_utils import ParticleTracker as ParticleTracker_cpp


class ParticleTracker:
    def __init__(self, T_step: float, N_humans: int, N_particles: int):
        """Initialize the ParticleTracker.

        :param T_step: double, step time of the tracker.
        :param N_humans: int, Maximum number of tracked humans in the tracker.
        :param N_particles: int, Number of particles to use in the tracker.
        :return: ParticleTracker object.
        """
        self.tracker = ParticleTracker_cpp(T_step, N_humans, N_particles)

    def add_observation(self, robot_perceptions):
        """Update the tracker based on a list of robot perceptions.

        :param robot_perceptions: list of dictionaries, each dictionary contains the position of a robot and a list of perceived humans.
        :return: list of edge probabilities for the tracker.
        """

        # self.merged_perceptions = ParticleTracker_cpp.merge_perceptions(robot_perceptions)
        # self.k_best_assignments = ParticleTracker_cpp.k_best_assignments(self.merged_perceptions, self.tracker.N_humans)

        individual_edge_probabilities = np.array(self.tracker.add_observation(robot_perceptions))
        return 1 - np.prod(1 - individual_edge_probabilities, axis=0)
        # return individual_edge_probabilities

    def merge_individual_edge_probabilities(self, individual_edge_probabilities):
        """Merge the edge probabilities of the tracker with the given edge probabilities.

        :param individual_edge_probabilities: List of list of edge probabilities to merge with the tracker.
        :return: List of merged edge probabilities for the tracker.
        """
        return 1 - np.prod(1 - np.array(individual_edge_probabilities), axis=0)

    def predict(self):
        """Predict the internal state of the tracker by T_step.

        :return: list of edge probabilities for the tracker.
        """
        individual_edge_probabilities = self.tracker.predict()
        return individual_edge_probabilities

    def save_training_data(self):
        """Save the collected training data of the tracker to a json file in the log folder."""
        self.tracker.save_training_data()
