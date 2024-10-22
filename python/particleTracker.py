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

    def add_observation(self, robot_perceptions) -> np.ndarray:
        """Update the tracker based on a list of robot perceptions.

        :param robot_perceptions: list of dictionaries, each dictionary contains the position of a robot and a list of perceived humans.
        :return: (N_edges,) np.ndarray of edge probabilities for the tracker.
        """
        return np.array(self.tracker.add_observation(robot_perceptions))

    def predict(self) -> np.ndarray:
        """Predict the internal state of the tracker by T_step.

        :return: (N_edges,) np.ndarray of edge probabilities for the tracker.
        """
        return np.array(self.tracker.predict())
