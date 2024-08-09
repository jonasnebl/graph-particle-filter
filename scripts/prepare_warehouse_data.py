import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import random
from plotter import Plotter
import os

# Load the graph data from the JSON file
with open('warehouse/graph_data.json', 'r') as f:
    graph_data = json.load(f)

nodes = graph_data['nodes']
edges = graph_data['edges']
edge_weights = [np.sqrt((nodes[edge[0]]['x'] - nodes[edge[1]]['x'])**2 + (nodes[edge[0]]['y'] - nodes[edge[1]]['y'])**2) for edge in edges]

# Load rack data from JSON file
with open('warehouse/rack_data.json', 'r') as f:
    rack_data = json.load(f)
polygons = rack_data["polygons"]

### --- Generate the C++ header file for the graph --- ###

# Generate the C++ header file content
header_content = """#ifndef GRAPH_DATA_H
#define GRAPH_DATA_H

#include <vector>

using Point = std::pair<double, double>;

namespace warehouse_data {
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
    std::vector<std::vector<Point>> node_polygons = {
"""

# Add node polygons to the header content
for node in nodes:
    header_content += "        {\n"
    for point in node['area']:
        header_content += f"            {{{point[0]}, {point[1]}}},\n"
    header_content += "        },\n"

header_content += """    };
}

#endif
"""

# Save the header content to a file
with open("src/warehouse_data.h", "w") as header_file:
    header_file.write(header_content)
os.system("clang-format -i -style=file src/warehouse_data.h")

### --- Visualize the warehouse --- ###

plotter = Plotter(show=True)