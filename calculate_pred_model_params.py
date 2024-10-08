from python.utils import load_warehouse_data_from_json
from python.constants import *
import json
from reliability.Fitters import Fit_Weibull_2P
from tqdm import tqdm
import numpy as np

nodes, edges, edge_weights, polygons, staging_nodes, storage_nodes, exit_nodes = load_warehouse_data_from_json()
successor_edges = [[i for i, next_edge in enumerate(edges) if edge[1] == next_edge[0]] for edge in edges]

with open(TRAINING_DATA_PATH, "r") as f:
    training_data = json.load(f)

pred_model_params = []
for i, edge in enumerate(edges):
    pred_model_params_one_edge = []
    relevant_samples = [sample for sample in training_data if sample[0] == i]

    # remove entries with no valid successor edge
    relevant_samples = [sample for sample in relevant_samples if sample[1] in successor_edges[sample[0]]]

    for successor_edge in successor_edges[i]:
        durations = [sample[2] for sample in relevant_samples if sample[1] == successor_edge]

        epsilon = 1
        select_edge_prob = (len(durations) + epsilon) / (len(relevant_samples) + epsilon * len(successor_edges[i]))

        if len(durations) >= 2:
            print(i, successor_edge, len(durations))
            # use MLE to fit a Weibull distribution to all data points
            # remove double entries to avoid singularity errors
            durations = [duration + 0.01 * np.random.normal() for duration in durations]
            fitted_weibull = Fit_Weibull_2P(failures=durations, show_probability_plot=False, print_results=False)
            alpha = fitted_weibull.alpha
            beta = fitted_weibull.beta
        else:
            # use the edge weights to estimate the distribution parameters
            alpha = 1.0
            beta = 1.0
        pred_model_params_one_edge.append([select_edge_prob, alpha, beta])

    pred_model_params.append(pred_model_params_one_edge)

with open(os.path.join(MODEL_PATH, "pred_model_params_new.json"), "w") as f:
    json.dump(pred_model_params, f, indent=4)
print("New prediction model parameters saved to pred_model_params_new.json")
