"""
Python wrapper for the C++ ParticleTracker class.
Enables documentation and type hints for the ParticleTracker class.
"""

import numpy as np
import sys

sys.path.append("build/")  # allos to import cpp_utils
from cpp_utils import ParticleTracker as ParticleTracker_cpp

sys.path.append("third-party/Murty")
from getkBestNoRankHung import getkBestNoRankHung


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

        # merged_perceptions = self.tracker.merge_perceptions(robot_perceptions)
        # robot_positions = merged_perceptions[0]
        # perceived_humans = merged_perceptions[1]
        # cost_matrix = np.array(self.tracker.calc_assignment_cost_matrix(perceived_humans))
        
        # assignment_proposals = self._assign_perceived_humans_to_tracks(cost_matrix)

        individual_edge_probabilities = np.array(self.tracker.add_observation(robot_perceptions))
        # return 1 - np.prod(1 - np.array(individual_edge_probabilities), axis=0)
        return individual_edge_probabilities

    def predict(self):
        """Predict the internal state of the tracker by T_step.

        :return: list of edge probabilities for the tracker.
        """
        individual_edge_probabilities = self.tracker.predict()
        return 1 - np.prod(1 - np.array(individual_edge_probabilities), axis=0)
    
    def _assign_perceived_humans_to_tracks(self, cost_matrix):
        """Assign perceived humans to internal humans.
        Uses Murty's algorithms to propose k best assignments.

        :param cost_matrix: np.array, cost matrix for the assignment problem.
        :return: list of assignment proposals.
        """
        cost_matrix = 1e-5 * cost_matrix.astype(np.float64)
        solutions, costs = getkBestNoRankHung(cost_matrix, 5)
        # print("Solutions:", solutions)
        print("Costs", costs)
        cols = solutions.argmax(axis=-1)    
        return -1