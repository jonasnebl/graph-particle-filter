import json
import numpy as np
from .constants import *

def load_warehouse_data_from_json():
    with open(GRAPH_PATH, 'r') as f:
        graph_data = json.load(f)

    nodes = graph_data['nodes']
    edges = graph_data['edges']
    edge_weights = [np.sqrt((nodes[edge[0]]['x'] - nodes[edge[1]]['x'])**2 + (nodes[edge[0]]['y'] - nodes[edge[1]]['y'])**2) for edge in edges]

    # Load rack data from JSON file
    with open(RACK_PATH, 'r') as f:
        rack_data = json.load(f)
    polygons = rack_data["polygons"]

    return nodes, edges, edge_weights, polygons