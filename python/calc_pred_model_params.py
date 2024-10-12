"""
Calculate the prediction model parameters.
"""

from utils import load_warehouse_data_from_json, get_successor_edges
from constants import *
import json
from reliability.Fitters import Fit_Weibull_2P
from tqdm import tqdm
import numpy as np
import pickle

nodes, edges, edge_weights, polygons, staging_nodes, storage_nodes, exit_nodes = load_warehouse_data_from_json()
successor_edges = get_successor_edges(edges)


def get_magic_training_data(log_filename):
    """Directly extracts the exact training data from a simulation log (magically).

    :param log_filename: Path to the simulation log
    :return: List of training data samples
    """
    with open(os.path.join(LOG_FOLDER, log_filename), "rb") as f:
        sim_log = pickle.load(f)
    sim_states = sim_log["sim_states"]
    T_step = sim_log["T_step"]
    N_humans = sim_log["N_humans"]
    N_robots = sim_log["N_robots"]

    # --- extract belonging edges ---
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
            if belonging_edges[j-1] != belonging_edges[j]:
                # save training data
                duration = simulation_time - edge_start_time
                training_data.append({
                    "previous_edge": previous_edge, 
                    "current_edge": belonging_edges[j-1], 
                    "next_edge": belonging_edges[j],
                    "duration": duration
                })

                # save values for next iteration
                edge_start_time = simulation_time
                previous_edge = belonging_edges[j-1]

    with open(TRAINING_DATA_PATH, "w") as f:
        json.dump(training_data, f, indent=4)

    print(f"{len(training_data)} training data samples saved to {TRAINING_DATA_PATH}")


def train():
    """Calculates the prediction model parameters for all edges and saves them to a JSON file."""

    with open(TRAINING_DATA_PATH, "r") as f:
        training_data = json.load(f)

    pred_model_params = []
    for i, edge in enumerate(edges):
        print(f"Calculating prediction model parameters for edge {i}")
        pred_model_params_one_edge = []
        relevant_samples = [sample for sample in training_data if sample["previous_edge"] == i]

        # remove entries with no valid successor edge
        relevant_samples = [sample for sample in relevant_samples if sample["current_edge"] in successor_edges[sample["previous_edge"]]]

        for successor_edge in successor_edges[i]:
            durations = [sample["duration"] for sample in relevant_samples if sample["current_edge"] == successor_edge]

            epsilon = 1
            select_edge_prob = (len(durations) + epsilon) / (len(relevant_samples) + epsilon * len(successor_edges[i]))

            while len(durations) < 10:
                # add dummy measurements based on the edge weights
                mean = edge_weights[successor_edge] / 1.3
                durations.append(np.clip(np.random.normal(mean, 0.5 * mean), 0.1 * mean, np.inf))

            # use MLE to fit a Weibull distribution to all data points
            # remove double entries to avoid singularity errors
            durations = [duration + 0.01 * np.random.normal() for duration in durations]
            fitted_weibull = Fit_Weibull_2P(failures=durations, show_probability_plot=False, print_results=False)
            alpha = fitted_weibull.alpha
            beta = fitted_weibull.beta
            pred_model_params_one_edge.append([select_edge_prob, alpha, beta])

        pred_model_params.append(pred_model_params_one_edge)

    with open(os.path.join(MODEL_PATH, "pred_model_params.json"), "w") as f:
        json.dump(pred_model_params, f, indent=4)
    print("New prediction model parameters saved to pred_model_params_new.json")


if __name__ == "__main__":
    get_magic_training_data("log_2024-10-11_21-06-28.pkl")
    train()
