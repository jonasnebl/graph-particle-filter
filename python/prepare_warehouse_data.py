"""
This script generates a C++ header file containing the warehouse graph data.
It also visualizes the warehouse graph with all node and edge indices
and saves the resulting figure to the figures folder.
"""

from plotter import Plotter
from utils import load_warehouse_data_from_json
import os

nodes, edges, edge_weights, polygons, staging_nodes, storage_nodes, exit_nodes = load_warehouse_data_from_json()

### --- Generate the C++ header file for the graph --- ###

# Generate the C++ header file content
header_content = """#ifndef WAREHOUSESIM_SRC_WAREHOUSE_DATA_H
#define WAREHOUSESIM_SRC_WAREHOUSE_DATA_H

#include <vector>

using Point = std::pair<double, double>;

namespace warehouse_data {
    inline std::vector<Point> nodes = {
"""

# Add nodes to the header content
for node in nodes:
    header_content += f"        {{{node['x']}, {node['y']}}},\n"

header_content += """    };
    inline std::vector<std::pair<int, int>> edges = {
"""

# Add edges to the header content
for edge in edges:
    header_content += f"        {{{edge[0]}, {edge[1]}}},\n"

header_content += """    };
    inline std::vector<double> edge_weights = {
"""

# Add edge weights to the header content
for weight in edge_weights:
    header_content += f"        {weight},\n"

header_content += """    };
    inline std::vector<std::vector<Point>> racks = {
"""

# Add racks (polygons) to the header content
for polygon in polygons:
    header_content += "        {\n"
    for point in polygon:
        header_content += f"            {{{point[0]}, {point[1]}}},\n"
    header_content += "        },\n"

header_content += """    };
    inline std::vector<int> staging_nodes = {
"""

# Add staging nodes to the header content
for node in staging_nodes:
    header_content += f"        {node},\n"

header_content += """    };
    inline std::vector<int> storage_nodes = {
"""

# Add storage nodes to the header content
for node in storage_nodes:
    header_content += f"        {node},\n"

header_content += """    };
    inline std::vector<int> exit_nodes = {
"""

# Add exit nodes to the header content
for node in exit_nodes:
    header_content += f"        {node},\n"

header_content += """    };
}

#endif
"""

# Save the header content to a file
with open("src/warehouse_data.h", "w") as header_file:
    header_file.write(header_content)
os.system("clang-format -i -style=file src/warehouse_data.h")

### --- Visualize the warehouse --- ###

plotter = Plotter(print_edge_indices=True)
plotter.savefig("warehouse.svg")
plotter.show()
