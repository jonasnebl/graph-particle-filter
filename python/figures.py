"""
Functions to generate figures for the final project report.
"""

import pickle
import os
import sys
from paths import *
import matplotlib.pyplot as plt
import matplotlib
import json
import numpy as np
from utils import load_warehouse_data_from_json, get_successor_edges
from evaluator import *

sys.path.append("build/")  # allos to import cpp_utils
from cpp_utils import Agent

matplotlib.use("TkAgg")

# plt.style.use('./figures/latex_matplotlib.mplstyle')
plt.style.use("default")


def plot_pred_model(edge):
    """
    Plot the prediction model for the given edge.
    """
    with open(os.path.join(MODEL_PATH, "pred_model_params.json"), "r") as f:
        pred_model_params = json.load(f)

    nodes, edges, edge_weights, polygons, staging_nodes, storage_nodes, exit_nodes = (
        load_warehouse_data_from_json()
    )
    successor_edges = get_successor_edges(edges)

    N_successors = len(pred_model_params[edge])
    fig, axs = plt.subplots(2, N_successors, figsize=(4 * N_successors, 7))
    fig.suptitle(f"Kante {edge}")

    for i, params in enumerate(pred_model_params[edge]):
        ax_dist = axs[0, i]
        ax_quiver = axs[1, i]

        select_edge_prob, alpha, beta = params
        x = np.linspace(0, 10, 1000)
        y = beta * x ** (beta - 1) / alpha**beta * np.exp(-((x / alpha) ** beta))
        ax_dist.plot(x, y)
        ax_dist.fill_between(x, y, alpha=0.2)
        ax_dist.set_title(f"Folgekante {i} (p={select_edge_prob:.2f})")
        ax_dist.set_xlabel("$t_{[e_i]}$ in Sekunden")
        ax_dist.set_ylabel("$p[t_{[e_i]}]$")

        # Add textbox with alpha and beta parameters
        textstr = f"$\\alpha={alpha:.2f}$\n$\\beta={beta:.2f}$"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax_dist.text(
            0.73,
            0.95,
            textstr,
            transform=ax_dist.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )

        # Quiver plot for the input edge and successor edges
        def update_lim(point, x_lim, y_lim):
            x_lim[0] = min(x_lim[0], point["x"] - 3)
            x_lim[1] = max(x_lim[1], point["x"] + 3)
            y_lim[0] = min(y_lim[0], point["y"] - 1)
            y_lim[1] = max(y_lim[1], point["y"] + 1)
            return x_lim, y_lim

        edge_start = nodes[edges[edge][0]]
        edge_end = nodes[edges[edge][1]]
        ax_quiver.quiver(
            edge_start["x"],
            edge_start["y"],
            edge_end["x"] - edge_start["x"],
            edge_end["y"] - edge_start["y"],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="blue",
            width=0.01,
            zorder=2,
        )
        x_lim, y_lim = update_lim(
            edge_start, [edge_start["x"], edge_start["x"]], [edge_end["y"], edge_end["y"]]
        )
        x_lim, y_lim = update_lim(edge_end, x_lim, y_lim)

        # display node
        ax_quiver.scatter(edge_end["x"], edge_end["y"], color="blue", zorder=3, s=100)

        for j, succ_edge in enumerate(successor_edges[edge]):
            succ_edge_start = nodes[edges[succ_edge][0]]
            succ_edge_end = nodes[edges[succ_edge][1]]
            x_lim, y_lim = update_lim(succ_edge_start, x_lim, y_lim)
            x_lim, y_lim = update_lim(succ_edge_end, x_lim, y_lim)
            color = "red" if j == i else "gray"
            ax_quiver.quiver(
                succ_edge_start["x"],
                succ_edge_start["y"],
                succ_edge_end["x"] - succ_edge_start["x"],
                succ_edge_end["y"] - succ_edge_start["y"],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
                width=0.03,
                zorder=1,
            )

        ax_quiver.set_xlim(x_lim)
        ax_quiver.set_ylim(y_lim)
        ax_quiver.set_aspect("equal")
        ax_quiver.axis("off")  # Remove the box around the quiver plot

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, f"pred_model_edge_{edge}.pdf"))
    plt.show()


def plot_results_multiple_thresholds(thresholds, false_negative_rates, cleared_edges_rates):
    """Plot the results for the ParticleTracker for multiple thresholds.

    :param thresholds: List of thresholds.
    :param false_negative_rates: List of false negative rates.
    :param cleared_edges_rates: List of cleared edges rates.
    """
    e = 1e-10
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for i in range(len(thresholds)):
        ax.scatter(
            cleared_edges_rates[i],
            false_negative_rates[i] + e,
            label="$p_{{lim}}=${:.10g}%".format(100 * thresholds[i]),
            marker="o",
        )
        # ax.annotate(annotation_text, (cleared_edges_rates[i], false_negative_rates[i]))
        if i > 0:
            ax.plot(
                [cleared_edges_rates[i - 1], cleared_edges_rates[i]],
                [false_negative_rates[i - 1] + e, false_negative_rates[i] + e],
                color="gray",
                zorder=-10,
            )
    ax.set_title(
        "Anteil von freigegebenen Kanten $r_{n}$ und davon \n "
        + "fälschlicherweise freigegebenen Kanten $r_{fn}$ für verschiedene Schwellwerte $p_{lim}$"
    )
    ax.set_xlabel("Anteil von freigegebenen Kanten $r_{n}$")
    ax.set_xlim([0, 1.01])
    ax.set_yscale("log")
    ax.grid(which="both")
    ax.set_ylabel("Anteil von fälschlicherweise freigegebenen Kanten $r_{fn}$")
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_yticks([1e-2, 1e-1, 1])
    ax.set_yticklabels(["{:,.2%}".format(x) for x in ax.get_yticks()])
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, "results_multiple_thresholds.pdf"))
    plt.show()


def plot_detection_probability():
    """Generate a plot of the detection probability."""
    distances = np.linspace(0, 25, 1000)
    detection_probabilities = np.array(
        [Agent.probability_in_viewrange(distance) for distance in distances]
    )
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(distances, detection_probabilities)
    ax.set_xlim([0, 25])
    ax.axhline(1.0, color="gray", linestyle="--")
    ax.axhline(0.0, color="gray", linestyle="--")
    ax.set_title("Detektionswahrscheinlichkeit $ p_{Detektion}(d) $ in Abhängigkeit der Entfernung")
    ax.set_xlabel("Entfernung $ d $ in Metern")
    ax.set_ylabel("Detektionswahrscheinlichkeit")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, "detection_probability.pdf"))
    plt.show()


def plot_N_humans_in_warehouse(filename: str):
    """Plot the number of humans in the warehouse over time."""
    with open(os.path.join(LOG_FOLDER, f"N_estimated_{filename}.pkl"), "rb") as f:
        N_estimated_log = np.array(pickle.load(f))
    with open(os.path.join(LOG_FOLDER, f"log_{filename}.pkl"), "rb") as f:
        sim_log = pickle.load(f)
    with open(os.path.join(LOG_FOLDER, f"N_tracks_{filename}.pkl"), "rb") as f:
        N_tracks_log = np.array(pickle.load(f))

    # determine true number of humans in the simulation
    N_humans_true = np.ones(N_estimated_log.shape) * sim_log["N_humans"]
    _, edges, _, _, _, _, exit_nodes = load_warehouse_data_from_json()
    for i, sim_state in enumerate(sim_log["sim_states"]):
        for agent in sim_state:
            if agent["type"] == "human" and agent["position"][0] > 62:  # 62 meters
                N_humans_true[i] -= 1

    timevec = np.arange(0, len(sim_log["sim_states"]) * sim_log["T_step"], sim_log["T_step"])
    timevec_minutes = timevec / 60

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(timevec_minutes, N_estimated_log.astype(np.float64) + 0.01, label="$N_{Schätzung}$")
    ax.plot(timevec_minutes, N_humans_true.astype(np.float64) - 0.02, label="$N_{wahr}$")
    ax.plot(timevec_minutes, N_tracks_log, label="$N_{Tracks}$")
    ax.legend()
    ax.set_title("Anzahl der wahrgenommenen Menschen im Lager")
    ax.set_xlabel("Zeit in Minuten")
    ax.set_xlim([0, timevec_minutes[-1]])
    ax.set_ylabel("Anzahl der wahrgenommenen Menschen")
    ax.set_ylim([0, max(N_tracks_log) + 0.1])
    ax.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, "N_estimated.pdf"))
    plt.show()


if __name__ == "__main__":
    # filename = "1hour_3_humans_4robots"
    filename = "3hour_4humans_4robots"

    # plot_detection_probability()

    # plot_pred_model(int(sys.argv[1]))

    # --- plot number of perceived humans comparison ---
    plot_N_humans_in_warehouse(filename)

    # --- Plot result metrics ---
    # thresholds = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    # false_negative_rates_human_centric, false_negative_rates_edge_centric, cleared_edges_rates = (
    #     evaluate_multiple_thresholds(thresholds, filename)
    # )
    # plot_results_multiple_thresholds(
    #     thresholds, false_negative_rates_human_centric, cleared_edges_rates
    # )
    # plot_results_multiple_thresholds(
    #     thresholds, false_negative_rates_edge_centric, cleared_edges_rates
    # )
