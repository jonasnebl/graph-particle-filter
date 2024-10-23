"""
This script sorts the nodes in the cartesian space by their y-coordinate first and then by their x-coordinate.
The graph itself doesn't change but sorting the nodes makes it easier to find nodes.
Additionally, it sorts the edges based on the mean of the edge
(average between start and end point).
"""

import json
from paths import *

# Load graph_data.json
with open(GRAPH_PATH, "r") as graph_file:
    graph_data = json.load(graph_file)

with open(NODE_MEANING_PATH, "r") as f:
    node_meanings = json.load(f)

# Order nodes by y first and then by x
range_vec = list(range(len(graph_data["nodes"])))
sort_mask = [
    index
    for _, index in sorted(
        zip(graph_data["nodes"], range_vec), key=lambda pair: 1000 * pair[0]["y"] + pair[0]["x"]
    )
]
graph_data["nodes"] = [graph_data["nodes"][i] for i in sort_mask]

# Replace node ids in edges with new ids
for edge in graph_data["edges"]:
    edge[0] = sort_mask.index(edge[0])
    edge[1] = sort_mask.index(edge[1])

# Replace node ids in node_meanings with new ids
for node in node_meanings["staging_nodes"]:
    node = sort_mask.index(node)
for node in node_meanings["storage_nodes"]:
    node = sort_mask.index(node)

# Calculate the mean position of each edge and sort edges based on the mean position
edges_with_mean = []
for edge in graph_data["edges"]:
    start_node = graph_data["nodes"][edge[0]]
    end_node = graph_data["nodes"][edge[1]]
    mean_x = (start_node["x"] + end_node["x"]) / 2
    mean_y = (start_node["y"] + end_node["y"]) / 2
    edges_with_mean.append((edge, mean_x, mean_y))

sorted_edges_with_mean = sorted(edges_with_mean, key=lambda item: (1000 * item[2] + item[1]))

# Update graph_data with sorted edges
graph_data["edges"] = [item[0] for item in sorted_edges_with_mean]

# Save updated values
with open(GRAPH_PATH, "w") as graph_file:
    json.dump(graph_data, graph_file, indent=4)

with open(NODE_MEANING_PATH, "w") as f:
    json.dump(node_meanings, f, indent=4)
