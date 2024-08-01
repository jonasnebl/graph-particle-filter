import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
matplotlib.use('TkAgg')
import json
import numpy as np

class Plotter:
    def __init__(self, show=False):

        with open('graph_data.json', 'r') as f:
            graph_data = json.load(f)
        self.nodes = graph_data['nodes']
        self.edges = graph_data['edges']
        self.node_positions = np.array([[node['x'], node['y']] for node in self.nodes])
        with open('rack_data.json', 'r') as f:
            rack_data = json.load(f)
        self.polygons = rack_data["polygons"]


        # Create a figure and axis
        self.fig, self.ax = plt.subplots()

        # Initialize scatter plots
        self.scat_graph = self.ax.scatter(self.node_positions[:, 0], self.node_positions[:, 1], s=100, c='skyblue', zorder=2)
        self.scat_agents = self.ax.scatter([], [], s=100, facecolor=[], zorder=0)
        # Plot the polygons for the racks
        for polygon in self.polygons:
            polygon_points = np.array(polygon)
            poly = Polygon(polygon_points, closed=True, fill=True, facecolor='gray', alpha=1, zorder=1)
            plt.gca().add_patch(poly)

        # Plot the edges
        for edge in self.edges:
            start, end = edge
            start_pos = self.node_positions[start]
            end_pos = self.node_positions[end]
            plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'b-', zorder=3)

        # Annotate the nodes with their indices
        for i, (x, y) in enumerate(self.node_positions):
            plt.text(x, y, str(i), fontsize=12, ha='right', color='black')

        # Plot the areas for each node in gray
        self.node_patches = []
        for node in self.nodes:
            if 'area' in node:
                area_points = np.array(node['area'])
                poly = Polygon(area_points, closed=True, fill=True, edgecolor='black', facecolor="gray", alpha=0.5, zorder=1)
                plt.gca().add_patch(poly)
                self.node_patches.append(poly)

        # List to keep track of perception-related elements
        self.perception_elements = []

        if show:
            plt.show()

    def update(self, state, node_probabilities):
        # Update the scatter plot
        positions = [(agent['x'], agent['y']) for agent in state]
        colors = ['blue' if agent['type'] == 'robot' else 'red' for agent in state]
        self.scat_agents.set_offsets(positions)
        self.scat_agents.set_color(colors)
        
        # Clear previous perceptions
        for elem in self.perception_elements:
            elem.remove()
        self.perception_elements.clear()

        # Visualize robot perceptions       
        for agent in state:
            if agent['type'] == 'robot' and 'perception' in agent:
                for perception in agent['perception']:
                    # Draw a thin black line from the robot to the perceived location
                    line = self.ax.plot([agent['x'], perception[0]], [agent['y'], perception[1]], 'k-', linewidth=0.5)
                    self.perception_elements.extend(line)

        # Update the colors of the polygons based on node_probabilities using a heatmap colormap
        cmap = plt.get_cmap('coolwarm')
        for i, node in enumerate(self.nodes):
            if 'area' in node:
                probability = node_probabilities[i]
                color = cmap(probability)
                self.node_patches[i].set_facecolor(color)

        plt.pause(1e-4)
