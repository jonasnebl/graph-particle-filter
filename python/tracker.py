import numpy as np
from .utils import load_warehouse_data_from_json
# from cpp_utils import get_belonging_node, get_observable_nodes


class Tracker:
    def __init__(self, N_robots):
        self.N_robots = N_robots
        self.nodes, self.edges, self.edge_weights, self.polygons = load_warehouse_data_from_json()
        self.N_edges = len(self.edges)

    def get_observable_nodes(self, robot_perceptions):
        """Get the observability of the nodes for all robots.

        returns: ndarray of shape (N_robots, N_nodes)
        """
        return np.array([robot["obversable_nodes"] for robot in robot_perceptions])

    def get_perceived_human_node_belongings(self, robot_perceptions):
        """Get the node belongings of the perceived humans for all robots.

        returns: ndarray of shape (N_robots, N_perceived_humans)
        """
        return np.array(
            [[human["belonging_node"] for human in robot["perceived_humans"]] for robot in robot_perceptions],
            dtype=int,
        )
