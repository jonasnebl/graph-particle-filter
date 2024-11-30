"""
This script generates a C++ header file containing the warehouse graph data.
It also visualizes the warehouse graph with all node and edge indices
and saves the resulting figure to the figures folder.
"""

from plotter import Plotter
from utils import load_warehouse_data_from_json, get_successor_edges
import os
import json
from paths import *
import numpy as np

nodes, edges, edge_weights, polygons, staging_nodes, storage_nodes, exit_nodes = (
    load_warehouse_data_from_json()
)

# Calculate successor edges
successor_edges = get_successor_edges(edges)

### --- Generate the C++ header file for the graph --- ###

# Generate the C++ header file content
header_content = """#ifndef WAREHOUSESIM_SRC_WAREHOUSE_DATA_H
#define WAREHOUSESIM_SRC_WAREHOUSE_DATA_H

#include <vector>

using Point = std::pair<double, double>;

struct graph_struct {
    std::vector<Point> nodes = {
"""

# Add nodes to the header content
for node in nodes:
    header_content += f"        {{{node['x']}, {node['y']}}},\n"

header_content += """    };
    std::vector<std::pair<int, int>> edges = {
"""

# Add edges to the header content
for edge in edges:
    header_content += f"        {{{edge[0]}, {edge[1]}}},\n"

header_content += """    };
    std::vector<double> edge_weights = {
"""

# Add edge weights to the header content
for weight in edge_weights:
    header_content += f"        {weight},\n"

header_content += """    };
    std::vector<std::vector<Point>> racks = {
"""

# Add racks (polygons) to the header content
for polygon in polygons:
    header_content += "        {\n"
    for point in polygon:
        header_content += f"            {{{point[0]}, {point[1]}}},\n"
    header_content += "        },\n"

header_content += """    };
    std::vector<int> staging_nodes = {
"""

# Add staging nodes to the header content
for node in staging_nodes:
    header_content += f"        {node},\n"

header_content += """    };
    std::vector<int> storage_nodes = {
"""

# Add storage nodes to the header content
for node in storage_nodes:
    header_content += f"        {node},\n"

header_content += """    };
    std::vector<int> exit_nodes = {
"""

# Add exit nodes to the header content
for node in exit_nodes:
    header_content += f"        {node},\n"

header_content += """    };
    std::vector<std::vector<int>> successor_edges = {
"""

# Add successor edges to the header content
for successors in successor_edges:
    header_content += "        {"
    header_content += ", ".join(map(str, successors))
    header_content += "},\n"

header_content += """    };
    std::vector<std::vector<double>> successor_edge_probabilities = {
"""

# Add successor edge probabilities to the header content
with open(os.path.join(MODEL_PATH, "successor_edge_probabilities.json"), "r") as f:
    successor_edge_probabilities = json.load(f)

for successor_edge_probabilities_one_edge in successor_edge_probabilities:
    header_content += "    {"
    header_content += ", ".join(map(str, successor_edge_probabilities_one_edge))
    header_content += "},\n"

header_content += """    };
    std::vector<std::vector<double>> duration_params = {
"""

# Add duration parameters to the header content
with open(os.path.join(MODEL_PATH, "duration_params.json"), "r") as f:
    duration_params = json.load(f)

for duration_params_one_edge in duration_params:
    header_content += "    {"
    header_content += ", ".join(map(str, duration_params_one_edge))
    header_content += "},\n"

header_content += """    };
    std::vector<std::vector<double>> N_perceived_likelihood_matrix = {
"""

# Add N perceived likelihood matrix to the header content
likelihood_matrix = np.loadtxt(N_HUMANS_LIKELIHOOD_MATRIX_PATH, delimiter=",")
for row in likelihood_matrix:
    header_content += "        {"
    header_content += ", ".join(map(str, row))
    header_content += "},\n"

header_content += """    };
};

#endif  // WAREHOUSESIM_SRC_WAREHOUSE_DATA_H
"""

# Save the header content to a file
with open("src/warehouse_data.h", "w") as header_file:
    header_file.write(header_content)
os.system("clang-format -i -style=file src/warehouse_data.h")

### --- Visualize the warehouse --- ###

plotter = Plotter(print_edge_indices=True)
plotter.savefig("warehouse.svg", format="svg")
plotter.savefig("warehouse.pdf", format="pdf")
plotter.show(blocking=False)
