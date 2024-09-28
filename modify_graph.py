import json

# Load graph_data.json
with open("warehouse/graph_data.json", "r") as graph_file:
    graph_data = json.load(graph_file)

# Load rack_data.json
with open("warehouse/rack_data.json", "r") as rack_file:
    rack_data = json.load(rack_file)

# N_old_nodes = len(graph_data["nodes"])
# new_nodes = []
# indices_new_nodes = []

# for i, node in enumerate(graph_data["nodes"]):
#     if node["y"] > 12.5:
#         new_nodes.append({"x": node["x"], "y": node["y"] + 10})
#         indices_new_nodes.append(i)
# graph_data["nodes"] += new_nodes

# new_edges = []
# for edge in graph_data["edges"]:
#     if edge[0] in indices_new_nodes and edge[1] in indices_new_nodes:
#         edge_start = indices_new_nodes.index(edge[0]) + N_old_nodes
#         edge_end = indices_new_nodes.index(edge[1]) + N_old_nodes
#         new_edges.append([edge_start, edge_end])
# graph_data["edges"] += new_edges

# Decrease and multiply cartesian values

# template_polygon = rack_data["polygons"][-1]

# new_polygons = []
# for i in range(0, 4):
#     for j in range(0, 3):
#         new_polygon = []
#         for point in template_polygon:
#             new_polygon.append([point[0] + i * 15, point[1] + j * 10])
#         new_polygons.append(new_polygon)

# rack_data["polygons"] += new_polygons


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


# Save updated graph_data.json
with open("warehouse/graph_data.json", "w") as graph_file:
    json.dump(graph_data, graph_file, indent=4)

# # Save updated rack_data.json
# with open("warehouse/rack_data.json", "w") as rack_file:
#     json.dump(rack_data, rack_file, indent=4)
