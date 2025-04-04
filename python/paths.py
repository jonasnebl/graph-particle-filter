"""
File for storing file paths as constants used in the project.
Also generates the necessary directories if they do not exist.
"""

import os

LOG_FOLDER = "logs"
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

MODEL_PATH = "models"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

WAREHOUSE_FOLDER = "warehouse"
if not os.path.exists(WAREHOUSE_FOLDER):
    os.makedirs(WAREHOUSE_FOLDER)

FIGURE_PATH = "figures"
if not os.path.exists(FIGURE_PATH):
    os.makedirs(FIGURE_PATH)

N_HUMANS_LIKELIHOOD_MATRIX_PATH = os.path.join(MODEL_PATH, "N_humans_likelihood_matrix.csv")
GRAPH_PATH = os.path.join(WAREHOUSE_FOLDER, "graph_data.json")
RACK_PATH = os.path.join(WAREHOUSE_FOLDER, "rack_data.json")
NODE_MEANING_PATH = os.path.join(WAREHOUSE_FOLDER, "node_meanings.json")


def duration_data_filename(use_magic_data: bool) -> str:
    """Return the filename for the duration data."""
    return "duration_data_magic.pkl" if use_magic_data else "duration_data.pkl"


def edge_change_data_filename(use_magic_data: bool) -> str:
    """Return the filename for the edge_change data."""
    return "edge_change_data_magic.pkl" if use_magic_data else "edge_change_data.pkl"
