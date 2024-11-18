"""
Calculate the prediction model parameters.
"""

from utils import load_warehouse_data_from_json, get_successor_edges
from paths import *
import json
from reliability.Fitters import Fit_Weibull_2P
from tqdm import tqdm
import numpy as np
import pickle
import time

nodes, edges, edge_weights, polygons, staging_nodes, storage_nodes, exit_nodes = (
    load_warehouse_data_from_json()
)
successor_edges = get_successor_edges(edges)


def get_belonging_edges(folder: str) -> tuple[list[list[int]], float]:
    """Extract the belonging edges from a simulation log.

    :param folder: Path to the folder with the simulation log
    :return: List of lists of belonging edges for each human
            Outer list is for the humans, inner list is for the time steps
            T_step is the time step of the simulation
    """
    start = time.time()
    with open(os.path.join(LOG_FOLDER, folder, "log.pkl"), "rb") as f:
        sim_log = pickle.load(f)
    sim_states = sim_log["sim_states"]
    T_step = sim_log["T_step"]
    N_humans = sim_log["N_humans"]
    N_robots = sim_log["N_robots"]
    print(f"Loaded simulation log in {time.time() - start:.2f} seconds.")

    # --- extract belonging edges ---
    start = time.time()
    human_belonging_edges = []
    for i in range(N_humans + N_robots):
        if sim_states[0][i]["type"] == "human":
            human_belonging_edges.append([state[i]["belonging_edge"] for state in sim_states])

    print(f"Extracted belonging edges in {time.time() - start:.2f} seconds.")

    return human_belonging_edges, T_step


def get_magic_edge_change_data(folder: str):
    """Directly extracts the edge change samples from a simulation log (magically).

    :param folder: Path to the folder with the simulation log
        Training data is a list of tuples (edge, next_edge)
    """
    human_belonging_edges, _ = get_belonging_edges(folder)

    # --- extract edge change training data from belonging edge vectors ---
    start = time.time()
    edge_change_training_data = []
    for belonging_edges in human_belonging_edges:
        for j in range(1, len(belonging_edges)):
            if belonging_edges[j - 1] != belonging_edges[j]:
                edge_change_training_data.append((belonging_edges[j - 1], belonging_edges[j]))
    print(
        f"Extracted {len(edge_change_training_data)} edge change samples in {time.time() - start:.2f} seconds."
    )

    # --- save generated edge change training data ---
    with open(os.path.join(LOG_FOLDER, folder, "edge_change_data_magic.pkl"), "wb") as f:
        pickle.dump(edge_change_training_data, f, pickle.HIGHEST_PROTOCOL)
    print(
        f"{len(edge_change_training_data)} edge change samples saved to {folder}/edge_change_data_magic.pkl."
    )


def get_magic_duration_data(folder: str):
    """Directly extracts the duration samples from a simulation log (magically).

    :param folder: Path to the folder with the simulation log
        Training data is a list of tuples (edge, duration)
    """
    belonging_edges, T_step = get_belonging_edges(folder)

    # --- extract duration training data from belonging edge vectors ---
    start = time.time()
    duration_training_data = []

    for belonging_edges_one_human in belonging_edges:
        simulation_time = 0
        last_time_changed = -1
        for j in range(1, len(belonging_edges_one_human)):
            if belonging_edges_one_human[j - 1] != belonging_edges_one_human[j]:
                if last_time_changed != -1:
                    duration_training_data.append(
                        (belonging_edges_one_human[j - 1], simulation_time - last_time_changed)
                    )
                last_time_changed = simulation_time
            simulation_time += T_step

    print(
        f"Extracted {len(duration_training_data)} duration samples in {time.time() - start:.2f} seconds."
    )

    # --- save generated training data ---
    with open(os.path.join(LOG_FOLDER, folder, "duration_data_magic.pkl"), "wb") as f:
        pickle.dump(duration_training_data, f, pickle.HIGHEST_PROTOCOL)
    print(
        f"{len(duration_training_data)} duration data samples saved to {folder}/duration_data_magic.pkl."
    )


def train_successor_edge_probabilities(folders: list[str], use_magic_data: bool = False):
    """Calculates the successor edge probabilities for all edges and saves them to a json file.

    :param folders: List of folder names containing the edge change training data.
        All the edge_change_training_data.pkl files in the folders
        will be used together to calculate the probabilities.
    """
    edge_change_training_data = []
    for folder in folders:
        with open(
            os.path.join(LOG_FOLDER, folder, edge_change_data_filename(use_magic_data)), "rb"
        ) as f:
            edge_change_training_data += pickle.load(f)

    successor_edge_probabilties = [
        [float(el) for el in list_] for list_ in successor_edges
    ]  # same shape as successor_edges but with float entries

    for i, edge_start_and_end_node in enumerate(edges):
        relevant_samples = [
            sample
            for sample in edge_change_training_data
            if sample[0] == i and sample[1] in successor_edges[i]
        ]

        # calculate the successor edge probabilities for every possible successor edge
        e = 0.01
        if len(relevant_samples) == 0:
            # no samples for this edge -> uniform distribution
            successor_edge_probabilties[i] = [
                1 / len(successor_edges[i]) for _ in range(len(successor_edges[i]))
            ]
        else:
            # samples available -> use them to calculate the probability
            for j, successor_edge in enumerate(successor_edges[i]):
                samples_where_successor_edge_was_taken = [
                    sample for sample in relevant_samples if sample[1] == successor_edge
                ]
                successor_edge_probabilties[i][j] = (
                    len(samples_where_successor_edge_was_taken) + e * len(relevant_samples)
                ) / (len(relevant_samples) * (1 + len(successor_edges[i]) * e))

        print(
            f"Calculated successor edge probabilities for edge {i} "
            + f"based on {len(relevant_samples)} samples."
        )

    with open(os.path.join(MODEL_PATH, "successor_edge_probabilities.json"), "w") as f:
        json.dump(successor_edge_probabilties, f, indent=4)


def train_durations(folders: list[str], use_magic_data: bool = False):
    """Calculates the duration params of the weibull distribution for all edges
        and saves them to a json file.

    :param folders: List of folder names containing the duration training data.
        All the duration_training_data.pkl files in the folders
        will be used together to calculate the probabilities.
    """
    duration_training_data = []
    for folder in folders:
        with open(
            os.path.join(LOG_FOLDER, folder, duration_data_filename(use_magic_data)), "rb"
        ) as f:
            duration_training_data += pickle.load(f)
    print(duration_training_data)

    start = time.time()
    duration_params = []
    for i in range(len(edges)):
        durations = [sample[1] for sample in duration_training_data if sample[0] == i]

        if len(durations) < 2:
            # not enough samples to fit a distribution
            duration_params.append([4.0, 1.5])
        else:
            # fit a Weibull distribution to the durations
            fitted_weibull = Fit_Weibull_2P(
                failures=durations, show_probability_plot=False, print_results=False
            )
            duration_params.append([fitted_weibull.alpha, fitted_weibull.beta])

        print(
            f"Calculated duration distribution parameters for edge {i} "
            + f"based on {len(durations)} samples. Time passed: {time.time() - start:.2f} seconds."
        )

    with open(os.path.join(MODEL_PATH, "duration_params.json"), "w") as f:
        json.dump(duration_params, f, indent=4)


def train_likelihood_matrix(folders: list[str]):
    """Trains the likelihood matrix for the number of humans estimation.

    :param folders: List of folder names containing the N_perceived.pkl files.
        The first folder corresponds to a simulation with 1 human,
        the second folder to a simulation with 2 humans, and so on.
        The size of the likelihood matrix will then be (len(folders)+1) x (len(folders)+1).
        The +1 is for the case where the true number of humans is 0.
        We don't need a folder for that, because N_perceived will always be 0.
    """
    N_tracks_max = len(folders)
    likelihood_matrix = np.zeros((N_tracks_max + 1, N_tracks_max + 1))
    likelihood_matrix[0, 0] = 1.0
    for i in range(1, N_tracks_max + 1):  # true number of humans
        with open(os.path.join(LOG_FOLDER, folders[i - 1], "N_perceived.pkl", "rb")) as f:
            N_perceived_log = pickle.load(f)
        for j in range(0, N_tracks_max + 1):  # perceived number of humans
            likelihood_matrix[i, j] = N_perceived_log.count(j) / len(N_perceived_log)
    print("Likelihood matrix:")
    print(likelihood_matrix)
    np.savetxt(N_HUMANS_LIKELIHOOD_MATRIX_PATH, likelihood_matrix, delimiter=",", fmt="%.6f")


if __name__ == "__main__":
    folder = "24h_4humans_4robots_100part"
    # get_magic_successor_edge_data(folder)
    get_magic_edge_change_data(folder)
    # get_magic_duration_data(folder)
    # train_durations([folder], use_magic_data=True)
    train_successor_edge_probabilities([folder], use_magic_data=True)
    # train_likelihood_matrix(
    #     [
    #         "1h_1humans_4robots_noleaving",
    #         "1h_2humans_4robots_noleaving",
    #         "1h_3humans_4robots_noleaving",
    #         "1h_4humans_4robots_noleaving",
    #         "1h_5humans_4robots_noleaving",
    #         "1h_6humans_4robots_noleaving",
    #         "1h_7humans_4robots_noleaving",
    #     ]
    # )
