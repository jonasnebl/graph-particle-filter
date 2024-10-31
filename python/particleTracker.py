"""
Python wrapper for the C++ ParticleTracker class.
Enables documentation and type hints for the ParticleTracker class.
"""

import numpy as np
import yaml
from utils import load_warehouse_data_from_json
import sys
from paths import *

sys.path.append("build/")  # allows to import cpp_utils
from cpp_utils import ParticleTracker as ParticleTracker_cpp


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


class ParticleTracker:
    def __init__(self, T_step: float, N_tracks_init: int, N_particles: int):
        """Initialize the ParticleTracker.

        :param T_step: double, step time of the tracker.
        :param N_tracks_init: int, Maximum number of tracked humans in the tracker.
        :param N_particles: int, Number of particles to use in the tracker.
        :return: ParticleTracker object.
        """
        self.tracker = ParticleTracker_cpp(T_step, N_tracks_init, N_particles)
        (
            self.nodes,
            self.edges,
            self.edge_weights,
            self.polygons,
            self.staging_nodes,
            self.storage_nodes,
            self.exit_nodes,
        ) = load_warehouse_data_from_json()

        # variables for the number of tracks
        self.simulation_time = 0
        self.T_step = T_step
        self.DEBOUNCE_TIME_SECONDS = 3 * 60
        self.last_time_track_added = -self.DEBOUNCE_TIME_SECONDS
        self.last_time_track_removed = -self.DEBOUNCE_TIME_SECONDS
        self.N_tracks = N_tracks_init
        self.N_perceived_humans_log = []
        self.N_humans_estimated_log = []
        self.N_tracks_log = []
        self.likelihood_matrix = np.loadtxt(N_HUMANS_LIKELIHOOD_MATRIX_PATH, delimiter=",")
        self.WINDOW_LENGTH_SECONDS = 10 * 60
        self.N_perceived_humans_window = np.array(
            [N_tracks_init] * int(self.WINDOW_LENGTH_SECONDS / 0.5), dtype=int
        )

    def add_observation(self, robot_perceptions) -> np.ndarray:
        """Update the tracker based on a list of robot perceptions.

        :param robot_perceptions: list of dictionaries, each dictionary contains the position of a robot and a list of perceived humans.
        :return: (N_edges,) np.ndarray of edge probabilities for the tracker.
        """
        perceived_humans, robot_positions = self.tracker.merge_perceptions(robot_perceptions)

        # handle number of tracks
        N_humans_estimated = self.tracker.estimate_N_humans(len(perceived_humans))
        self.N_humans_estimated_log.append(N_humans_estimated)
        N_tracks_new = N_humans_estimated + 1  # add a safety margin
        if (
            N_tracks_new > self.N_tracks
            and self.simulation_time - self.last_time_track_removed >= self.DEBOUNCE_TIME_SECONDS
        ):
            self.tracker.add_one_track()
            self.last_time_track_added = self.simulation_time
            self.N_tracks += 1
        elif (
            N_tracks_new < self.N_tracks
            and self.simulation_time - self.last_time_track_added >= self.DEBOUNCE_TIME_SECONDS
        ):
            self.tracker.remove_one_track()
            self.last_time_track_removed = self.simulation_time
            self.N_tracks -= 1
        self.N_tracks_log.append(self.N_tracks)

        return np.array(self.tracker.add_merged_perceptions(perceived_humans, robot_positions))

    def predict(self) -> np.ndarray:
        """Predict the internal state of the tracker by T_step.

        :return: (N_edges,) np.ndarray of edge probabilities for the tracker.
        """
        self.simulation_time += self.T_step
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
