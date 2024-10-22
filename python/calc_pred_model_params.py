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

nodes, edges, edge_weights, polygons, staging_nodes, storage_nodes, exit_nodes = load_warehouse_data_from_json()
successor_edges = get_successor_edges(edges)


def get_magic_training_data(log_filename):
    """Directly extracts the exact training data from a simulation log (magically).

    :param log_filename: Path to the simulation log
    :return: List of training data samples
    """
    start = time.time()
    with open(os.path.join(LOG_FOLDER, log_filename), "rb") as f:
        sim_log = pickle.load(f)
    sim_states = sim_log["sim_states"]
    T_step = sim_log["T_step"]
    N_humans = sim_log["N_humans"]
    N_robots = sim_log["N_robots"]
    print(f"Loaded simulation log in {time.time() - start:.2f} seconds.")

    # --- extract belonging edges ---
    start = time.time()
    human_belonging_edges = []
    i = 0
    for i in range(N_humans + N_robots):
        if sim_states[0][i]["type"] == "human":
            human_belonging_edges.append([state[i]["belonging_edge"] for state in sim_states])
            i += 1

    # --- extract training data from belonging edge vectors ---
    training_data = []
    for i, belonging_edges in enumerate(human_belonging_edges):
        simulation_time = 0
        edge_start_time = 0
        previous_edge = belonging_edges[0]
        for j in range(2, len(belonging_edges) - 1):
            simulation_time += T_step
            if belonging_edges[j - 1] != belonging_edges[j]:
                # save training data
                duration = simulation_time - edge_start_time
                training_data.append(
                    {
                        "previous_edge": previous_edge,
                        "current_edge": belonging_edges[j - 1],
                        "next_edge": belonging_edges[j],
                        "duration": duration,
                    }
                )

                # save values for next iteration
                edge_start_time = simulation_time
                previous_edge = belonging_edges[j - 1]
    print(f"Extracted {len(training_data)} training data samples in {time.time() - start:.2f} seconds.")

    with open(TRAINING_DATA_PATH, "w") as f:
        json.dump(training_data, f, indent=4)

    print(f"{len(training_data)} training data samples saved to {TRAINING_DATA_PATH}")


def train_successor_edge_probabilities():
    """Calculates the successor edge probabilities for all edges and saves them to a json file."""

    with open(TRAINING_DATA_PATH, "r") as f:
        training_data = json.load(f)

    successor_edge_probabilties = [
        [float(el) for el in list_] for list_ in successor_edges
    ]  # same shape as successor_edges but with float entries

    for i, edge_start_and_end_node in enumerate(edges):
        relevant_samples = [sample for sample in training_data if sample["current_edge"] == i]

        # remove entries with no valid successor edge
        relevant_samples = [
            sample for sample in relevant_samples if sample["next_edge"] in successor_edges[sample["current_edge"]]
        ]

        # calculate the successor edge probabilities for every possible successor edge
        e = 0.01
        if len(relevant_samples) == 0:
            # no samples for this edge -> uniform distribution
            successor_edge_probabilties[i] = [1 / len(successor_edges[i]) for _ in range(len(successor_edges[i]))]
        else:
            # samples available -> use them to calculate the probability
            for j, successor_edge in enumerate(successor_edges[i]):
                samples_where_successor_edge_was_taken = [
                    sample for sample in relevant_samples if sample["next_edge"] == successor_edge
                ]
                successor_edge_probabilties[i][j] = (
                    len(samples_where_successor_edge_was_taken) + e * len(relevant_samples)
                ) / (len(relevant_samples) * (1 + len(successor_edges[i]) * e))

        print(f"Calculated successor edge probabilities for edge {i} " + f"based on {len(relevant_samples)} samples.")

    with open(os.path.join(MODEL_PATH, "successor_edge_probabilities.json"), "w") as f:
        json.dump(successor_edge_probabilties, f, indent=4)


if __name__ == "__main__":
    get_magic_training_data("24hours_10humans_1robot_0.5seconds.pkl")
    train_successor_edge_probabilities()
