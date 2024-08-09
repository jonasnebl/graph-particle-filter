import json

# Load graph_data.json
with open('warehouse/graph_data.json', 'r') as graph_file:
    graph_data = json.load(graph_file)

# Load rack_data.json
with open('warehouse/rack_data.json', 'r') as rack_file:
    rack_data = json.load(rack_file)

# Decrease and multiply cartesian coordinates
for node in graph_data['nodes']:
    node['x'] = (node['x'] - 1) * 10
    node['y'] = (node['y'] - 1) * 10
    node['area'] = [[(point[0]-1) * 10, (point[1]-1)*10] for point in node['area']]

# Decrease and multiply cartesian values
polygons_resized = []
for polygon in rack_data['polygons']:
    polygons_resized.append([[(point[0]-1) * 10, (point[1]-1)*10] for point in polygon])
rack_data['polygons'] = polygons_resized

# Save updated graph_data.json
with open('warehouse/graph_data.json', 'w') as graph_file:
    json.dump(graph_data, graph_file, indent=4)

# Save updated rack_data.json
with open('warehouse/rack_data.json', 'w') as rack_file:
    json.dump(rack_data, rack_file, indent=4)