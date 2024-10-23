import numpy as np
from particleTracker import ParticleTracker
from utils import load_warehouse_data_from_json


def calc_false_negative_rate(
    edge_probabilities_log: np.ndarray, clear_threshold: float, sim_log: list
) -> float:
    """Calculate the false negative rate based on the edge probabilities.
    The false negative rate is the rate of cleared but occupied edges
    to the total number of cleared_edges

    :param edge_probabilities: (N_timesteps, N_edges) np.ndarray of edge probabilities.
    :param clear_threshold: float, Threshold for clearing an edge.
    :param sim_log: list of simulation states
    :return: float, False negative rate.
    """
    true_occupied_edges = np.zeros(shape=edge_probabilities_log.shape, dtype=bool)
    for i, sim_state in enumerate(sim_log["sim_states"]):
        human_belonging_edges = [
            agent["belonging_edge"] for agent in sim_state if agent["type"] == "human"
        ]
        true_occupied_edges[i] = [
            edge in human_belonging_edges for edge in range(true_occupied_edges.shape[1])
        ]
    cleared_edges_log = calc_cleared_edges_log(edge_probabilities_log, clear_threshold)

    # divide by cleared edges rate to only consider the false negatives when the edges are cleared
    cleared_edges_rate = calc_cleared_edges_rate(edge_probabilities_log, clear_threshold)
    return (
        np.mean(np.logical_and(cleared_edges_log, true_occupied_edges).astype(np.float64))
        / cleared_edges_rate
    )


def calc_cleared_edges_rate(edge_probabilites_log: np.ndarray, clear_threshold: float) -> float:
    """Calculate the rate of cleared edges based on the edge probabilities.

    :param edge_probabilities: (N_timesteps, N_edges) np.ndarray of edge probabilities.
    :param clear_threshold: float, Threshold for clearing an edge.
    :return: float, Rate of cleared edges.
    """
    cleared_edges_log = calc_cleared_edges_log(edge_probabilites_log, clear_threshold)
    return np.mean(cleared_edges_log.astype(np.float64))


def calc_cleared_edges_log(
    edge_probabilities_log: np.ndarray, clear_threshold: float
) -> np.ndarray:
    """Calculate the rate of cleared edges based on the edge probabilities.

    :param edge_probabilities: (N_timesteps, N_edges) np.ndarray of edge probabilities.
    :param clear_threshold: float, Threshold for clearing an edge.
    :return: (N_timesteps, N_edges ) np.ndarray of cleared edges.
    """
    _, edges, _, _, _, _, _ = load_warehouse_data_from_json()
    cleared_edges = np.zeros(shape=edge_probabilities_log.shape, dtype=bool)
    for i, edge_probabilities in enumerate(edge_probabilities_log):
        cleared_edges[i, :] = ParticleTracker.static_get_cleared_edges(
            edge_probabilities, clear_threshold, edges
        )
    return cleared_edges
