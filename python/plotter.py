import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
matplotlib.use('TkAgg')
import json
import numpy as np
from moviepy.editor import VideoClip, ImageSequenceClip
from moviepy.video.io.bindings import mplfig_to_npimage
from datetime import datetime
from .constants import *
import os

class Plotter:
    def __init__(self, show=False, record_frames=False):
        """Plotter to plot robot and human movement in the warehouse

        :param show: Show the warehouse structure after initializing the tracker object. Blocking!
        :param record_frames: Record one frame every time update() is called. Slows down the plotter heavily.
        :returns: Plotter object
        """

        self.record_frames = record_frames

        with open(GRAPH_PATH, 'r') as f:
            graph_data = json.load(f)
        self.nodes = graph_data['nodes']
        self.edges = graph_data['edges']
        self.node_positions = np.array([[node['x'], node['y']] for node in self.nodes])
        with open(RACK_PATH, 'r') as f:
            rack_data = json.load(f)
        self.polygons = rack_data["polygons"]


        # Create a figure and axis
        self.fig, self.ax = plt.subplots()

        self.ax.set_title('Warehouse Simulation')
        self.ax.set_xlabel('X in m')
        self.ax.set_ylabel('Y in m')

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

        # List to store frames to later generate a video
        if self.record_frames:
            self.frames = []

        if show:
            plt.show()

    def update(self, state, node_probabilities):
        # Update the scatter plot
        positions = [agent["ego_position"] for agent in state]
        colors = ['blue' if agent['type'] == 'robot' else 'red' for agent in state]
        self.scat_agents.set_offsets(positions)
        self.scat_agents.set_color(colors)
        
        # Clear previous perceptions
        for elem in self.perception_elements:
            elem.remove()
        self.perception_elements.clear()

        # Visualize robot perceptions       
        for agent in state:
            if agent['type'] == 'robot' and 'perceived_humans' in agent:
                for perceived_human in agent['perceived_humans']:
                    # Draw a thin black line from the robot to the perceived location
                    line = self.ax.plot([agent['ego_position'][0], perceived_human['pos_mean'][0]], 
                                        [agent['ego_position'][1], perceived_human['pos_mean'][1]], 'k-', linewidth=0.5)
                    self.perception_elements.extend(line)

        # Update the colors of the polygons based on node_probabilities using a heatmap colormap
        cmap = plt.get_cmap('coolwarm')
        for i, node in enumerate(self.nodes):
            if 'area' in node:
                probability = node_probabilities[i]
                color = cmap(probability)
                self.node_patches[i].set_facecolor(color)

        # Capture the current frame
        if self.record_frames:
            self.frames.append(mplfig_to_npimage(self.fig))

        plt.pause(1e-4)

    def create_video(self, T_step):
        if not self.record_frames:
            raise Exception("You must set option record_frames=True to create a video.")
        else:
            fps = int(1 / T_step)
            clip = ImageSequenceClip(self.frames, fps=fps)
            clip.write_videofile(os.path.join(LOG_FOLDER, 
                'video_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.mp4', fps=fps))
