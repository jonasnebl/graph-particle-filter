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
    N_sim_states = len(sim_log["sim_states"])
    human_belonging_edges = np.zeros((N_sim_states, sim_log["N_humans"]), dtype=int)
    for i, sim_state in enumerate(sim_log["sim_states"]):
        human_belonging_edges[i] = np.array(
            [agent["belonging_edge"] for agent in sim_state if agent["type"] == "human"]
        )
    cleared_edges_log = calc_cleared_edges_log(edge_probabilities_log, clear_threshold)

    is_the_human_belonging_edge_cleared = cleared_edges_log[
        np.arange(N_sim_states)[:, None], human_belonging_edges
    ]

    return np.mean(is_the_human_belonging_edge_cleared.astype(np.float64))


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
