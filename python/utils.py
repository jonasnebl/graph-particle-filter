import json
import numpy as np
from paths import *


def load_warehouse_data_from_json() -> (
    tuple[
        list[dict],  # nodes
        list[tuple[int, int]],  # edges
        list[float],  # edge_weights
        list[list[list[float]]],  # polygons
        list[int],  # staging_nodes
        list[int],  # storage_nodes
        list[int],  # exit_nodes
    ]
):
    """Loads the warehouse data from the JSON files.

    :return: nodes, edges, edge_weights, polygons, staging_nodes, storage_nodes, exit_nodes
    """
    with open(GRAPH_PATH, "r") as f:
        graph_data = json.load(f)

    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    edge_weights = [
        np.sqrt(
            (nodes[edge[0]]["x"] - nodes[edge[1]]["x"]) ** 2
            + (nodes[edge[0]]["y"] - nodes[edge[1]]["y"]) ** 2
        )
        for edge in edges
    ]

    # Load rack data from JSON file
    with open(RACK_PATH, "r") as f:
        rack_data = json.load(f)
    polygons = rack_data["polygons"]

    # Load staging and storage nodes
    with open(NODE_MEANING_PATH, "r") as f:
        node_meanings = json.load(f)
    staging_nodes = node_meanings["staging_nodes"]
    storage_nodes = node_meanings["storage_nodes"]
    exit_nodes = node_meanings["exit_nodes"]

    return nodes, edges, edge_weights, polygons, staging_nodes, storage_nodes, exit_nodes


def get_successor_edges(edges: list[tuple[int, int]]) -> list[list[int]]:
    """Get the successor edges for each edge.

    :param edges: list of list of int, List of edges.
    :return: list of list of int, List of successor edges.
    """
    return [[i for i, next_edge in enumerate(edges) if edge[1] == next_edge[0]] for edge in edges]
