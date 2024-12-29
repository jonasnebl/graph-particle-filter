"""
Functions to generate figures for the final project report.
"""

import argparse
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
from math import gamma

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
    fig.suptitle(f"Wahrscheinlichkeiten für Folgekanten von Kante {edge}", fontsize=18)

    # make sure that the displayed probabilities sum up to 1 despite rounding
    successor_edge_probabilities_one_edge = [
        np.round(p, decimals=3) for p in successor_edge_probabilities[edge]
    ]
    successor_edge_probabilities_one_edge[-1] = 1 - sum(successor_edge_probabilities_one_edge[:-1])

    for i, successor_edge_probability in enumerate(successor_edge_probabilities_one_edge):
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
    plt.show(block=False)


def plot_results_multiple_thresholds(
    thresholds: list[float],
    false_negative_rates_list: list[list[float]],
    list_legends: list[str],
    cleared_edges_rates: list[float],
):
    """Plot the results for the ParticleTracker for multiple thresholds.

    :param thresholds: List of thresholds.
    :param false_negative_rates: List of list of false negative rates.
    :param cleared_edges_rates: List of cleared edges rates.
    """

    print("Thresholds: ", thresholds)
    for i, false_negative_rates in enumerate(false_negative_rates_list):
        print(
            f"False negative rates {list_legends[i]}: {', '.join(f'{100 * rate:.7g}%' for rate in false_negative_rates)}"
        )
    print(f"Cleared edges rates: {', '.join(f'{100 * rate:.7g}%' for rate in cleared_edges_rates)}")

    e = 1e-10  # add e becasue 0 cannot be displayed on a log scale
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    line_colors = ["blue", "orange", "green", "red", "purple", "brown"]
    point_colors = ["brown", "blue", "orange", "green", "purple", "lime"]
    lines = []
    points = []
    for i, false_negative_rates in enumerate(false_negative_rates_list):
        for j in range(len(thresholds)):
            scatter_point = ax.scatter(
                cleared_edges_rates[j],
                false_negative_rates[j] + e,
                marker="o",
                zorder=10,
                color=point_colors[j],
            )
            if i == 0:
                points.append(scatter_point)
            # ax.annotate(annotation_text, (cleared_edges_rates[i], false_negative_rates[i]))
            if j > 0:
                lines.append(
                    ax.plot(
                        [cleared_edges_rates[j - 1], cleared_edges_rates[j]],
                        [false_negative_rates[j - 1] + e, false_negative_rates[j] + e],
                        color=line_colors[i],
                        zorder=-10,
                    )
                )
    ax.set_title("Negativ-Rate $r_{n}$ und Falsch-Negativ-Rate $r_{fn}$", fontsize=18)
    ax.set_xlabel("Negativ-Rate $r_{n}$", fontsize=14)
    ax.set_xlim([0, 1.01])
    ax.set_yscale("log")
    ax.grid(which="both")
    ax.set_ylabel("Falsch-Negativ-Rate $r_{fn}$", fontsize=14)
    ax.set_ylim(ax.get_ylim()[::-1])
    # ax.set_yticks([1e-2, 1e-1, 1])
    ax.set_yticklabels(["{:g}%".format(100 * x) for x in ax.get_yticks()])

    ax.legend(
        points,
        ["$p_{{lim}}=${:.10g}%".format(100 * threshold) for threshold in thresholds],
        loc="lower left",
    )
    ax2 = ax.twinx()
    for i, legend in enumerate(list_legends):
        ax2.plot(np.NaN, np.NaN, label=legend, color=line_colors[i])
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, "results_multiple_thresholds.pdf"))
    plt.show(block=False)


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
    ax.set_title("Detektionswahrscheinlichkeit $ p_{Detektion}(d) $ eines Menschen", fontsize=14)
    ax.set_xlabel("Entfernung $ d $ in Metern")
    ax.set_ylabel("$ p_{Detektion}(d) $")
    ax.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, "detection_probability.pdf"))
    plt.show(block=False)


def plot_N_humans_in_warehouse(folder: str, specifier: str = ""):
    """Plot the number of humans in the warehouse over time.

    :param folder: Folder name of the simulation to be used.
    :param specifier: Additional specifier for the filename under which the figure is saved.
    """
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

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(timevec_minutes, N_estimated_log.astype(np.float64) + 0.01, label="$N_{humans,est}$")
    ax.plot(timevec_minutes, N_humans_true.astype(np.float64) - 0.02, label="$N_{wahr}$")
    ax.plot(timevec_minutes, N_tracks_log, label="$N_{Tracks}$")
    ax.legend(fontsize=12)
    ax.set_title("Schätzung der Anzahl der Menschen im Lager", fontsize=18)
    ax.set_xlabel("Zeit in Minuten", fontsize=14)
    ax.set_xlim([0, timevec_minutes[-1]])
    ax.set_ylabel("$ N_{humans} $", fontsize=14)
    ax.set_ylim([0, max(N_tracks_log) + 0.1])
    ax.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, "N_estimated" + specifier + ".pdf"))
    plt.show(block=False)


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
    filename = "edge_change_distribution_magic" if use_magic_data else "edge_change_distribution"
    plotter.savefig(filename, format="svg")
    plotter.show(blocking=False)


def plot_model_difference():
    """Plot the relative difference distribution between two models."""
    with open(os.path.join(MODEL_PATH, "successor_edge_probabilities.json"), "rb") as f:
        successor_edge_probabilities = json.load(f)
    with open(os.path.join(MODEL_PATH, "successor_edge_probabilities_magic.json"), "rb") as f:
        successor_edge_probabilities_magic = json.load(f)
    with open(os.path.join(MODEL_PATH, "duration_params.json"), "rb") as f:
        duration_params = json.load(f)
    with open(os.path.join(MODEL_PATH, "duration_params_magic.json"), "rb") as f:
        duration_params_magic = json.load(f)

    # --- edge change model differences ---
    mean_differences = []
    for probabilities1, probabilities2 in zip(
        successor_edge_probabilities, successor_edge_probabilities_magic
    ):
        mean_differences.append(
            np.mean(np.abs(np.array(probabilities1) - np.array(probabilities2)))
        )

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    axs[0].hist(mean_differences, bins=30)
    axs[0].set_xlabel("$\Delta \\overline{p}_i$")
    axs[0].set_ylabel("Wahrscheinlichkeitsdichte")
    axs[0].set_title("Nachfolgekanten")
    axs[0].grid()

    # --- duration model differences ---
    expected_value = lambda alpha, beta: alpha * gamma(1 + 1 / beta)
    standard_deviation = lambda alpha, beta: alpha**2 * (
        gamma(1 + 2 / beta) - gamma(1 + 1 / beta) ** 2
    )

    expected_value_differences = []
    standard_deviation_differences = []
    for (alpha, beta), (alpha_magic, beta_magic) in zip(duration_params, duration_params_magic):
        expected_value_difference = expected_value(alpha, beta) - expected_value(
            alpha_magic, beta_magic
        )
        # ignore outliers that correspond to the out of warehouse edge
        if np.abs(expected_value_difference) < 100:
            expected_value_differences.append(expected_value_difference)
        standard_deviation_differences.append(
            standard_deviation(alpha, beta) - standard_deviation(alpha_magic, beta_magic)
        )

    axs[1].hist(expected_value_differences, bins=30)
    axs[1].set_xlabel("$\Delta E[t_{[e_i]}]$ in Sekunden")
    axs[1].set_title("Aufenthaltszeiten")
    axs[1].set_ylabel("Wahrscheinlichkeitsdichte")
    axs[1].grid()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_PATH, "model_differences.pdf"))
    plt.show(block=False)


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
    filename = "duration_distribution_magic" if use_magic_data else "duration_distribution"
    plotter.savefig(filename, format="svg")
    plotter.show(blocking=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_humans_folder_short", type=str)
    parser.add_argument("--N_humans_folder_long", type=str)
    parser.add_argument("--training_folder", type=str)
    parser.add_argument("--results_folder1", type=str)
    parser.add_argument("--results_folder2", type=str)
    parser.add_argument("--results_folder3", type=str)
    parser.add_argument("--results_folder4", type=str)
    args = parser.parse_args()

    # --- figures for chapters 1-3 ---
    plot_detection_probability()
    plot_edge_change_model(25)

    # -- plot number of perceived humans comparison ---
    plot_N_humans_in_warehouse(args.N_humans_folder_short, "_5min")
    plot_N_humans_in_warehouse(args.N_humans_folder_long, "_10min")

    # --- plot training data distributions and comparisons ---
    plot_edge_change_data_distribution(args.training_folder, use_magic_data=True, scale=5)
    plot_edge_change_data_distribution(args.training_folder, scale=5)
    plot_duration_data_distribution(args.training_folder, use_magic_data=True, scale=0.05)
    plot_duration_data_distribution(args.training_folder, scale=0.05)
    plot_model_difference()

    # --- Plot overall result metrics ---
    thresholds = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    false_negative_rates_edge_centric = np.zeros_like(thresholds)
    false_negative_rates_human_centric = np.zeros_like(thresholds)
    cleared_edges_rates = np.zeros_like(thresholds)
    for folder in [
        args.results_folder1,
        args.results_folder2,
        args.results_folder3,
        args.results_folder4,
    ]:
        (
            false_negative_rates_human_centric_folder_i,
            false_negative_rates_edge_centric_folder_i,
            cleared_edges_rates_folder_i,
        ) = evaluate_multiple_thresholds(thresholds, folder)
        false_negative_rates_human_centric += 0.25 * np.array(
            false_negative_rates_human_centric_folder_i
        )
        false_negative_rates_edge_centric += 0.25 * np.array(
            false_negative_rates_edge_centric_folder_i
        )
        cleared_edges_rates += 0.25 * np.array(cleared_edges_rates_folder_i)

    plot_results_multiple_thresholds(
        thresholds,
        [false_negative_rates_human_centric, false_negative_rates_edge_centric],
        ["Mensch-zentriert", "Kanten-zentriert"],
        cleared_edges_rates,
    )
