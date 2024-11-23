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
from plotter import Plotter

sys.path.append("build/")  # allos to import cpp_utils
from cpp_utils import Agent

matplotlib.use("TkAgg")

# plt.style.use('./figures/latex_matplotlib.mplstyle')
plt.style.use("default")


def plot_edge_change_model(edge, use_magic_data: bool = False):
    """
    Plot the prediction model for the given edge.
    """
    filename = (
        "successor_edge_probabilities_magic.json"
        if use_magic_data
        else "successor_edge_probabilities.json"
    )
    with open(os.path.join(MODEL_PATH, filename), "r") as f:
        successor_edge_probabilities = json.load(f)

    nodes, edges, _, _, _, _, _ = load_warehouse_data_from_json()
    successor_edges = get_successor_edges(edges)

    N_successors = len(successor_edge_probabilities[edge])
    fig, axs = plt.subplots(1, N_successors, figsize=(2.5 * N_successors, 2.5))
    fig.suptitle(f"Wahrscheinlichkeit für Folgekanten von Kante {edge}")

    for i, successor_edge_probability in enumerate(successor_edge_probabilities[edge]):
        ax = axs[i]

        # Quiver plot for the input edge and successor edges
        def update_lim(point, x_lim, y_lim):
            x_lim[0] = min(x_lim[0], point["x"] - 0.5)
            x_lim[1] = max(x_lim[1], point["x"] + 0.5)
            y_lim[0] = min(y_lim[0], point["y"] - 0.5)
            y_lim[1] = max(y_lim[1], point["y"] + 0.5)
            return x_lim, y_lim

        edge_start = nodes[edges[edge][0]]
        edge_end = nodes[edges[edge][1]]
        ax.quiver(
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
        ax.scatter(edge_end["x"], edge_end["y"], color="black", zorder=3, s=20)

        for j, succ_edge in enumerate(successor_edges[edge]):
            succ_edge_start = nodes[edges[succ_edge][0]]
            succ_edge_end = nodes[edges[succ_edge][1]]
            x_lim, y_lim = update_lim(succ_edge_start, x_lim, y_lim)
            x_lim, y_lim = update_lim(succ_edge_end, x_lim, y_lim)
            color = "red" if j == i else "gray"
            ax.quiver(
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

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect("equal")
        ax.set_title("$p={:.1f}$%".format(100 * successor_edge_probability))
        ax.axis("off")  # Remove the box around the quiver plot

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, f"edge_change_model_{edge}.pdf"))
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
    # ax.set_yticks([1e-2, 1e-1, 1])
    ax.set_yticklabels(["{:g}%".format(100 * x) for x in ax.get_yticks()])
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


def plot_N_humans_in_warehouse(folder: str):
    """Plot the number of humans in the warehouse over time."""
    with open(os.path.join(LOG_FOLDER, folder, "N_estimated.pkl"), "rb") as f:
        N_estimated_log = np.array(pickle.load(f))
    with open(os.path.join(LOG_FOLDER, folder, "log.pkl"), "rb") as f:
        sim_log = pickle.load(f)
    with open(os.path.join(LOG_FOLDER, folder, "N_tracks.pkl"), "rb") as f:
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

    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
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


def plot_edge_change_data_distribution(
    folder: str, use_magic_data: bool = False, scale: float = -1
):
    """Plot the spatial distribution of the training data in the warehouse.

    :param folder: Folder name of the simulation to be used.
    :param scale: float, scale the size of the displayed dots, -1: auto scale
    """
    nodes, edges, _, _, _, _, _ = load_warehouse_data_from_json()
    with open(
        os.path.join(LOG_FOLDER, folder, edge_change_data_filename(use_magic_data)), "rb"
    ) as f:
        edge_change_data = pickle.load(f)

    edge_change_data_distribution_edges = [
        len([sample for sample in edge_change_data if sample[0] == i]) for i in range(len(edges))
    ]
    edge_change_data_distribution = []
    for i in range(len(nodes)):
        edge_change_data_distribution.append(0)
        for j in range(len(edges)):
            if edges[j][1] == i:
                edge_change_data_distribution[-1] += edge_change_data_distribution_edges[j]

    plotter = Plotter()
    plotter.node_weight_plot(
        edge_change_data_distribution,
        title="Edge change data distribution\nbased on {} samples".format(len(edge_change_data)),
        scale=scale,
    )
    filename = (
        "edge_change_distribution_magic.pdf" if use_magic_data else "edge_change_distribution.pdf"
    )
    plotter.savefig(filename, format="pdf")
    plotter.show(blocking=True)


def plot_edge_change_model_difference(filename1, filename2):
    """Plot the relative difference distribution between two models.

    :param filename1: json filename of the successor_edge_probabilities of the first model
    :param filename2: json filename of the successor_edge_probabilities of the second model
    """
    with open(os.path.join(MODEL_PATH, filename1), "rb") as f:
        successor_edge_probabilities1 = json.load(f)
    with open(os.path.join(MODEL_PATH, filename2), "rb") as f:
        successor_edge_probabilities2 = json.load(f)

    mean_differences = []
    for i, (probabilities1, probabilities2) in enumerate(
        zip(successor_edge_probabilities1, successor_edge_probabilities2)
    ):
        mean_differences.append(
            np.mean(np.abs(np.array(probabilities1) - np.array(probabilities2)))
        )
    plotter = Plotter()
    plotter.node_weight_plot(
        mean_differences, title="Distribution of differences between edge change models"
    )
    plotter.savefig("edge_change_model_difference.pdf", format="pdf")
    plotter.show(blocking=True)


def plot_duration_data_distribution(folder: str, use_magic_data: bool = False, scale: float = -1):
    """Plot the spatial distribution of the training data in the warehouse.

    :param folder: Folder name of the simulation to be used.
    :param scale: float, scale the width of the displayed lines, -1: auto scale
    """
    _, edges, _, _, _, _, _ = load_warehouse_data_from_json()
    with open(os.path.join(LOG_FOLDER, folder, duration_data_filename(use_magic_data)), "rb") as f:
        duration_data = pickle.load(f)

    duration_data_distribution = [
        len([sample for sample in duration_data if sample[0] == i]) for i in range(len(edges))
    ]

    plotter = Plotter()
    plotter.edge_weight_plot(
        duration_data_distribution,
        title="Edge change data distribution\nbased on {} samples".format(len(duration_data)),
        scale=scale,
    )
    filename = "duration_distribution_magic.pdf" if use_magic_data else "duration_distribution.pdf"
    plotter.savefig(filename, format="pdf")
    plotter.show(blocking=True)


if __name__ == "__main__":
    folder = "24h_4humans_4robots_100part"

    # plot_edge_change_data_distribution(folder, use_magic_data=True, scale=8)
    # plot_edge_change_data_distribution(folder, scale=8)
    # plot_duration_data_distribution(folder, use_magic_data=True, scale=0.1)
    # plot_duration_data_distribution(folder, scale=0.1)

    # plot_edge_change_model_difference(
    #     "successor_edge_probabilities.json", "successor_edge_probabilities_magic.json"
    # )

    # plot_detection_probability()

    plot_edge_change_model(int(sys.argv[1]))

    # --- plot number of perceived humans comparison ---
    # plot_N_humans_in_warehouse(folder)

    # # --- Plot result metrics ---
    # thresholds = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    # false_negative_rates_human_centric, false_negative_rates_edge_centric, cleared_edges_rates = (
    #     evaluate_multiple_thresholds(thresholds, folder)
    # )
    # plot_results_multiple_thresholds(
    #     thresholds, false_negative_rates_human_centric, cleared_edges_rates
    # )
    # plot_results_multiple_thresholds(
    #     thresholds, false_negative_rates_edge_centric, cleared_edges_rates
    # )
