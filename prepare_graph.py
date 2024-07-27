import json
import matplotlib.pyplot as plt
import numpy as np

# Load the graph data from the JSON file
with open('graph_data.json', 'r') as f:
    graph_data = json.load(f)

nodes = graph_data['nodes']
edges = graph_data['edges']
edge_weights = graph_data['edge_weights']

### --- Generate the C++ header file --- ###

# Generate the C++ header file content
header_content = """#ifndef GRAPH_DATA_H
#define GRAPH_DATA_H

#include <vector>
namespace graph_data {
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
}

#endif
"""

# Write the header content to a file
with open('src/graph_data.h', 'w') as f:
    f.write(header_content)

print("C++ header file 'graph_data.h' generated successfully.")



### --- Visualize the graph --- ###

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

plt.title('Graph Visualization')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
