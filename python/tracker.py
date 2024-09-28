import numpy as np
from .utils import load_warehouse_data_from_json
# from cpp_utils import get_belonging_node, get_observable_nodes


class Tracker:
    def __init__(self, N_robots):
        self.N_robots = N_robots
        self.nodes, self.edges, self.edge_weights, self.polygons, self.staging_nodes, self.storage_nodes = (
            load_warehouse_data_from_json()
        )
        self.N_edges = len(self.edges)

    def get_perceived_human_node_belongings(self, robot_perceptions):
        """Get the node belongings of the perceived humans for all robots.

        returns: ndarray of shape (N_robots, N_perceived_humans)
        """
        return np.array(
            [[human["belonging_edge"] for human in robot["perceived_humans"]] for robot in robot_perceptions],
            dtype=int,
        )

    # def distance_of_point_to_edge(self, position, p1, p2):
    #     # Calculate the minimum distance from a point to a line segment
    #     x0, y0 = position
    #     x1, y1 = p1
    #     x2, y2 = p2
    #     dx = x2 - x1
    #     dy = y2 - y1
    #     if np.all((dx == 0) & (dy == 0)):
    #         return np.hypot(x0 - x1, y0 - y1)
    #     t = np.clip(((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy), 0, 1)
    #     nearest_x = x1 + t * dx
    #     nearest_y = y1 + t * dy
    #     return np.hypot(x0 - nearest_x, y0 - nearest_y)

    # def get_perceived_human_node_belongings(self, robot_perceptions):
    #     positions = np.array([perception["position"] for perception in robot_perceptions])
    #     headings = np.array([perception["heading"] for perception in robot_perceptions])

    #     p1 = self.nodes[self.edges[:, 0]]
    #     p2 = self.nodes[self.edges[:, 1]]

    #     # Calculate distances and heading differences for all edges
    #     distances = np.zeros((len(robot_perceptions), len(self.edges)))
    #     for i, (position, heading) in enumerate(zip(positions, headings)):
    #         cartesian_distances = self.distance_of_point_to_edge(position, p1, p2)
    #         edge_headings = np.arctan2(p2[:, 1] - p1[:, 1], p2[:, 0] - p1[:, 0])
    #         heading_differences = np.abs(heading - edge_headings)
    #         heading_differences = np.minimum(heading_differences, 2 * np.pi - heading_differences)
    #         distances[i] = cartesian_distances + 2 * heading_differences

    #     # Find the index of the closest edge for each perception
    #     closest_edges = np.argmin(distances, axis=1)
    #     return closest_edges
