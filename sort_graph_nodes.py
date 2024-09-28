import json
from python.constants import *

# Load graph_data.json
with open(GRAPH_PATH, "r") as graph_file:
    graph_data = json.load(graph_file)

with open(NODE_MEANING_PATH, "r") as f:
    node_meanings = json.load(f)

# order nodes by y first and then by x
range_vec = list(range(len(graph_data["nodes"])))
sort_mask = [
    index
    for _, index in sorted(zip(graph_data["nodes"], range_vec), key=lambda pair: 1000 * pair[0]["y"] + pair[0]["x"])
]
graph_data["nodes"] = [graph_data["nodes"][i] for i in sort_mask]

for edge in graph_data["edges"]:
    edge[0] = sort_mask.index(edge[0])
    edge[1] = sort_mask.index(edge[1])

for node in node_meanings["staging_nodes"]:
    node["node_id"] = sort_mask.index(node["node_id"])
for node in node_meanings["storage_nodes"]:
    node["node_id"] = sort_mask.index(node["node_id"])


# Save updated values
with open(GRAPH_PATH, "w") as graph_file:
    json.dump(graph_data, graph_file, indent=4)

with open(NODE_MEANING_PATH, "w") as f:
    json.dump(node_meanings, f, indent=4)
