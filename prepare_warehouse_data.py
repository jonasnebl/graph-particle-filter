import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import random

# Load the graph data from the JSON file
with open('graph_data.json', 'r') as f:
    graph_data = json.load(f)

nodes = graph_data['nodes']
edges = graph_data['edges']
edge_weights = [np.sqrt((nodes[edge[0]]['x'] - nodes[edge[1]]['x'])**2 + (nodes[edge[0]]['y'] - nodes[edge[1]]['y'])**2) for edge in edges]
# edge_weights = graph_data['edge_weights']

# load rack data from JSON file
with open('rack_data.json', 'r') as f:
    rack_data = json.load(f)
polygons = rack_data["polygons"]



### --- Generate the C++ header file for the graph --- ###

# Generate the C++ header file content
header_content = """#ifndef GRAPH_DATA_H
#define GRAPH_DATA_H

#include <vector>
namespace warehouse_data {
    std::vector<std::pair<double, double>> nodes = {
"""

# Add nodes to the header content
for node in nodes:
    header_content += f"        {{{node['x']}, {node['y']}}},\n"

header_content += """    };
    std::vector<std::pair<std::size_t, std::size_t>> edges = {
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
    std::vector<std::vector<std::pair<double, double>>> racks = {
"""

# Add racks (polygons) to the header content
for polygon in polygons:
    header_content += "        {\n"
    for point in polygon:
        header_content += f"            {{{point[0]}, {point[1]}}},\n"
    header_content += "        },\n"

header_content += """    };
}

#endif
"""

# Save the header content to a file
with open("src/warehouse_data.h", "w") as header_file:
    header_file.write(header_content)




### --- Visualize the warehouse --- ###

# Extract node positions
node_positions = np.array([(node['x'], node['y']) for node in nodes])

# Create the plot
plt.figure(figsize=(8, 6))

# Plot the nodes
plt.scatter(node_positions[:, 0], node_positions[:, 1], s=100, c='skyblue', zorder=2)

# Plot the edges
for edge in edges:
    start, end = edge
    start_pos = node_positions[start]
    end_pos = node_positions[end]
    plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'b-', zorder=1)

# Annotate the nodes with their indices
for i, (x, y) in enumerate(node_positions):
    plt.text(x, y, str(i), fontsize=12, ha='right', color='black')

# Plot the polygons for the racks
for polygon in polygons:
    polygon_points = np.array(polygon)
    poly = Polygon(polygon_points, closed=True, fill=True, edgecolor='r', facecolor='lightcoral', alpha=0.5, zorder=0)
    plt.gca().add_patch(poly)

# Plot the areas for each node in random shades of gray
for node in nodes:
    if 'area' in node:
        area_points = np.array(node['area'])
        poly = Polygon(area_points, closed=True, fill=True, edgecolor='black', facecolor="gray", alpha=0.5, zorder=1)
        plt.gca().add_patch(poly)

plt.title('Warehouse Visualization')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
