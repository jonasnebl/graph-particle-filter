import numpy as np
from particleTracker import ParticleTracker
from utils import load_warehouse_data_from_json
import os
import pickle
from paths import *


def calc_false_negative_human_centric(
    edge_probabilities_log: np.ndarray, clear_threshold: float, sim_log: list
) -> float:
    """Calculate the false negative rate based on the edge probabilities.
    The false negative rate is the rate of cleared but occupied edges
    to the total number of cleared_edges from the standpoint of the humans
    in the simulation.

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


def calc_false_negative_edge_centric(
    edge_probabilities_log: np.ndarray, clear_threshold: float, sim_log: list
) -> float:
    """Calculate the false negative rate based on the edge probabilities.
    The false negative rate is the rate of cleared but occupied edges
    to the total number of cleared_edges from the standpoint of the edges.
    Generally much lower than the human-centric false negative rate,
    because of the low prior probability of a human being on a specific edge.

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

    is_a_human_on_the_cleared_edge = np.zeros_like(cleared_edges_log, dtype=bool)
    for i in range(N_sim_states):
        for j in range(is_a_human_on_the_cleared_edge.shape[1]):
            if j in human_belonging_edges[i] and cleared_edges_log[i, j]:
                is_a_human_on_the_cleared_edge[i, j] = True

    return np.mean(is_a_human_on_the_cleared_edge.astype(np.float64)) / np.mean(
        cleared_edges_log.astype(np.float64)
    )


def calc_cleared_edges_rate(edge_probabilites_log: np.ndarray, clear_threshold: float) -> float:
    """Calculate the rate of cleared edges based on the edge probabilities.

    :param edge_probabilities: (N_timesteps, N_edges) np.ndarray of edge probabilities.
    :param clear_threshold: float, Threshold for clearing an edge.
    :return: float, Rate of cleared edges.
    """
    cleared_edges_log = calc_cleared_edges_log(edge_probabilites_log, clear_threshold)
    return np.mean(cleared_edges_log.astype(np.float64))


def evaluate_multiple_thresholds(thresholds: list[float], folder: str):
    """Calculate all metrics for a list of multiple thresholds.

    :param thresholds: list of float, Thresholds for clearing an edge.
    :param filename: str, Filename of the log files.
    """
    # load edge_probabilities and sim_log
    with open(os.path.join(LOG_FOLDER, folder, "edge_probabilities.pkl"), "rb") as f:
        edge_probabilities_log = pickle.load(f)
    with open(os.path.join(LOG_FOLDER, folder, "log.pkl"), "rb") as f:
        sim_log = pickle.load(f)

    false_negative_rates_human_centric = []
    false_negative_rates_edge_centric = []
    cleared_edges_rates = []
    for threshold in thresholds:
        false_negative_human_centric = calc_false_negative_human_centric(
            np.array(edge_probabilities_log), threshold, sim_log
        )
        false_negative_edge_centric = calc_false_negative_edge_centric(
            np.array(edge_probabilities_log), threshold, sim_log
        )
        cleared_edges_rate = calc_cleared_edges_rate(np.array(edge_probabilities_log), threshold)

        false_negative_rates_human_centric.append(false_negative_human_centric)
        false_negative_rates_edge_centric.append(false_negative_edge_centric)
        cleared_edges_rates.append(cleared_edges_rate)
        print(
            "Threshold {:.10g}%:  fn_human_centric={:.5f}%  fn_edge_centric={:.5}%  cleared_edges={:.1f}%".format(
                100 * threshold,
                100 * false_negative_human_centric,
                100 * false_negative_edge_centric,
                100 * cleared_edges_rate,
            )
        )
    return (
        false_negative_rates_human_centric,
        false_negative_rates_edge_centric,
        cleared_edges_rates,
    )


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


if __name__ == "__main__":
    thresholds = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    false_negative_rates_human_centric, false_negative_rates_edge_centric, cleared_edges_rates = (
        evaluate_multiple_thresholds(thresholds, filename="2024-10-23_19-23-37")
    )
