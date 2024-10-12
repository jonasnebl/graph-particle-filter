"""
Functions to generate figures for the final project report.
"""

import sys
import os
from constants import *
import matplotlib.pyplot as plt
import matplotlib
import json
import numpy as np
from utils import load_warehouse_data_from_json, get_successor_edges

matplotlib.use("TkAgg")

# plt.style.use('./figures/latex_matplotlib.mplstyle')
plt.style.use("default")


def plot_pred_model(edge):
    """
    Plot the prediction model for the given edge.
    """
    with open(os.path.join(MODEL_PATH, "pred_model_params.json"), "r") as f:
        pred_model_params = json.load(f)

    nodes, edges, edge_weights, polygons, staging_nodes, storage_nodes, exit_nodes = load_warehouse_data_from_json()
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
        ax_dist.text(0.73, 0.95, textstr, transform=ax_dist.transAxes, fontsize=12, verticalalignment="top", bbox=props)

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
        x_lim, y_lim = update_lim(edge_start, [edge_start["x"], edge_start["x"]], [edge_end["y"], edge_end["y"]])
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


if __name__ == "__main__":
    plot_pred_model(int(sys.argv[1]))
