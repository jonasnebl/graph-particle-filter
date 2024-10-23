"""
Python wrapper for the C++ ParticleTracker class.
Enables documentation and type hints for the ParticleTracker class.
"""

import numpy as np
import yaml
from utils import load_warehouse_data_from_json
import sys

sys.path.append("build/")  # allos to import cpp_utils
from cpp_utils import ParticleTracker as ParticleTracker_cpp


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


class ParticleTracker:
    def __init__(self, T_step: float, N_humans: int, N_particles: int):
        """Initialize the ParticleTracker.

        :param T_step: double, step time of the tracker.
        :param N_humans: int, Maximum number of tracked humans in the tracker.
        :param N_particles: int, Number of particles to use in the tracker.
        :return: ParticleTracker object.
        """
        self.tracker = ParticleTracker_cpp(T_step, N_humans, N_particles)
        (
            self.nodes,
            self.edges,
            self.edge_weights,
            self.polygons,
            self.staging_nodes,
            self.storage_nodes,
            self.exit_nodes,
        ) = load_warehouse_data_from_json()

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

    @staticmethod
    def static_get_cleared_edges(
        probabilities: np.ndarray, clear_threshold: float, edges: list
    ) -> np.ndarray:
        """Return an array of edges that are considered cleared based on a very low probability.
        Both the edge and its opposing edge must have a probability below the clear_threshold.

        :param probabilities: (N_edges,) np.ndarray of edge probabilities.
        :param edges: list of edges as tuples of start and end node index
        :return: (N_edges,) np.ndarray of cleared edge probabilities.
        """
        opposing_edge_probabilities = np.array(
            [probabilities[edges.index([edge[1], edge[0]])] for edge in edges]
        )
        cleared_edges = np.zeros_like(probabilities, dtype=bool)
        for i in range(len(probabilities)):
            if (
                probabilities[i] < clear_threshold
                and opposing_edge_probabilities[i] < clear_threshold
            ):
                cleared_edges[i] = True  # edge is cleared
            else:
                cleared_edges[i] = False  # edge is not cleared
        return cleared_edges

    def get_cleared_edges(self, probabilities: np.ndarray) -> np.ndarray:
        """Non-static wrapper for static_get_cleared_edges

        :param probabilities: (N_edges,) np.ndarray of edge probabilities.
        :return: (N_edges,) np.ndarray of cleared edge probabilities.
        """
        clear_threshold = config["clear_threshold"]
        return self.static_get_cleared_edges(probabilities, clear_threshold, self.edges)
