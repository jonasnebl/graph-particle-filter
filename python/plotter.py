import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon

matplotlib.use("TkAgg")
import json
import numpy as np
import pickle
from moviepy.editor import VideoClip, ImageSequenceClip
from moviepy.video.io.bindings import mplfig_to_npimage
from datetime import datetime
from paths import *
import os


class Plotter:
    def __init__(self, print_probabilites=False, print_edge_indices=False, clear_threshold=1e-3):
        """Plotter to plot robot and human movement in the warehouse

        :param show: Show the warehouse structure after initializing the tracker object. Blocking!
        :param record_frames: Record one frame every time update() is called. Slows down the plotter heavily.
        :returns: Plotter object
        """

        self.print_probabilites = print_probabilites
        self.clear_threshold = clear_threshold

        with open(GRAPH_PATH, "r") as f:
            graph_data = json.load(f)
        self.nodes = graph_data["nodes"]
        self.edges = graph_data["edges"]
        self.node_positions = np.array([[node["x"], node["y"]] for node in self.nodes])
        with open(RACK_PATH, "r") as f:
            rack_data = json.load(f)
        self.polygons = rack_data["polygons"]

        # Create a figure and axis
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(24, 16)

        self.ax.set_title("Warehouse Simulation", fontsize=24)
        self.ax.set_xlabel("X in m")
        self.ax.set_ylabel("Y in m")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.axis("off")

        # Initialize scatter plots
        self.scat_graph = self.ax.scatter(
            self.node_positions[:, 0], self.node_positions[:, 1], s=50, c="skyblue", zorder=2
        )
        self.scat_agents = self.ax.scatter([], [], s=200, facecolor=[], zorder=19)
        # Plot the polygons for the racks
        for polygon in self.polygons:
            polygon_points = np.array(polygon)
            poly = Polygon(
                polygon_points, closed=True, fill=True, facecolor="gray", alpha=1, zorder=1
            )
            plt.gca().add_patch(poly)

        # Annotate the nodes with their indices
        for i, (x, y) in enumerate(self.node_positions):
            plt.text(x, y, str(i), fontsize=12, ha="right", color="black")

        # show the edges
        for edge in self.edges:
            start, end = edge
            start_pos = self.node_positions[start]
            end_pos = self.node_positions[end]
            plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], "black", zorder=10)

        # Annotate the edges with their indices
        if print_edge_indices:
            self.annotate_edges(np.arange(len(self.edges), dtype=int))

        # List to keep track of perception-related elements
        self.perception_elements = []
        self.green_lines = []

        # List to store frames to later generate a video
        self.frames = []

    def reset(self):
        """Clear previous arrows and text annotations"""
        for elem in self.perception_elements:
            elem.remove()
        self.perception_elements.clear()
        for patch in self.ax.patches:
            if isinstance(patch, matplotlib.patches.FancyArrow):
                patch.remove()
        for green_line in self.green_lines:
            green_line[0].remove()
        self.green_lines = []
        for text in self.ax.texts:
            text.remove()

    def update_sim_state(self, state):
        """Update the warehouse plot with the current state of the simulation

        :param state: List of dictionaries containing the state of each agent
        """
        # Update the scatter plot
        positions = [agent["position"] for agent in state]
        colors = ["orange" if agent["type"] == "robot" else "green" for agent in state]
        self.scat_agents.set_offsets(positions)
        self.scat_agents.set_color(colors)

        # Visualize robot perceptions
        for agent in state:
            if agent["type"] == "robot" and "perceived_humans" in agent:
                for perceived_human in agent["perceived_humans"]:
                    # Draw a thin black line from the robot to the perceived location
                    line = self.ax.plot(
                        [agent["position"][0], perceived_human["position"][0]],
                        [agent["position"][1], perceived_human["position"][1]],
                        "b-",
                        linewidth=0.5,
                        zorder=20,  # Set a higher zorder for perception lines
                    )
                    self.perception_elements.extend(line)
                    perception_scat = self.ax.scatter(
                        perceived_human["position"][0],
                        perceived_human["position"][1],
                        color="b",
                        marker="x",
                        zorder=20,  # Set a higher zorder for perception points
                    )
                    self.perception_elements.append(perception_scat)

    def update_edge_probabilities(self, edge_probabilities):
        """Displays the edge probabilities for each tracked human individually

        :param edge_probabilities: List of edge probabilities for each human
        """
        # --- Update the edges with arrows based on edge_probabilities ---
        cmap = plt.get_cmap("coolwarm")
        for i, edge in enumerate(self.edges):
            start, end = edge
            start_pos = self.node_positions[start]
            end_pos = self.node_positions[end]
            probability = edge_probabilities[i]
            nonlinearity = 1 / 3  # value < 1 to make low probabilities more distinguishable
            color = cmap(probability**nonlinearity)
            linewidth = 1 + 19 * probability**nonlinearity  # Line width ranges from 1 to 25
            alpha = (
                0.1 if probability == 0 else 1
            )  # Make arrow almost invisible if probability is zero
            arrow = matplotlib.patches.FancyArrow(
                start_pos[0],
                start_pos[1],
                end_pos[0] - start_pos[0],
                end_pos[1] - start_pos[1],
                width=0.05 * linewidth,  # Adjust the width of the arrow
                color=color,
                alpha=alpha,
                length_includes_head=True,
                head_width=0.1 * linewidth,  # Adjust the head width of the arrow
                head_length=0.2 * linewidth,  # Adjust the head length of the arrow
                zorder=2,  # Set a lower zorder for arrows
            )
            self.ax.add_patch(arrow)

        if self.print_probabilites:
            self.annotate_edges(
                ["{:.3f}".format(probability) for probability in edge_probabilities]
            )

    def update_cleared_edges(self, cleared_edges: list[bool]):
        """Highlight the edges that are considered cleared based on a very low probability.

        :param cleared_edges: List of bool cleared edge probabilities
        """
        for i, edge_cleared in enumerate(cleared_edges):
            if edge_cleared:
                start_pos = self.node_positions[self.edges[i][0]]
                end_pos = self.node_positions[self.edges[i][1]]
                self.green_lines.append(
                    plt.plot(
                        [start_pos[0], end_pos[0]],
                        [start_pos[1], end_pos[1]],
                        "lime",
                        zorder=-10,
                        linewidth=20,
                    )
                )

    def update_individual_edge_probabilities(
        self, individual_edge_probabilities: list[list[float]]
    ):
        """Displays the edge probabilities for each tracked human individually

        :param individual_edge_probabilities: List of list of edge probabilities for each human
        """
        colors = ["red", "blue", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]
        if len(individual_edge_probabilities) >= len(colors):
            raise ValueError("We don't have enough colors to display so many humans.")

        for i, edge in enumerate(self.edges):
            for j, edge_probabilities in enumerate(individual_edge_probabilities):
                start, end = edge
                start_pos = self.node_positions[start]
                end_pos = self.node_positions[end]
                probability = edge_probabilities[i]
                linewidth = 1 + 19 * probability  # Line width ranges from 1 to 20
                alpha = (
                    0.1 if probability == 0 else 1
                )  # Make arrow almost invisible if probability is zero
                arrow = matplotlib.patches.FancyArrow(
                    start_pos[0],
                    start_pos[1],
                    end_pos[0] - start_pos[0],
                    end_pos[1] - start_pos[1],
                    width=0.05 * linewidth,  # Adjust the width of the arrow
                    color=colors[j],
                    alpha=alpha,
                    length_includes_head=True,
                    head_width=0.1 * linewidth,  # Adjust the head width of the arrow
                    head_length=0.2 * linewidth,  # Adjust the head length of the arrow
                    zorder=int(-probability * 100),  # Set a lower zorder for arrows
                )
                self.ax.add_patch(arrow)

        if self.print_probabilites:
            max_edge_probabilities = np.max(individual_edge_probabilities, axis=0)
            self.annotate_edges(
                ["{:.3f}".format(max_probability) for max_probability in max_edge_probabilities]
            )

    def show(self, blocking: bool = True):
        """Show the current state of the warehouse plot.

        :param blocking: If True, the plot will be blocking, otherwise non-blocking.
        """
        plt.tight_layout()
        if blocking:
            plt.show()
        else:
            plt.pause(1e-4)

    def capture_frame(self):
        """Capture the current frame and store it for later video generation"""
        self.frames.append(mplfig_to_npimage(self.fig))

    def create_video(self, T_step: float, speed: float = 1.0):
        if len(self.frames) == 0:
            raise Exception("You must capture frames with capture_frame() before creating a video.")
        else:
            fps = speed / T_step
            clip = ImageSequenceClip(self.frames, fps=fps)
            clip.write_videofile(
                os.path.join(
                    LOG_FOLDER,
                    "video_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mp4",
                ),
                fps=fps,
            )

    def savefig(self, filename: str, format: str = "svg"):
        """Save the currently displayed warehouse as a pdf figure"""
        if format not in ["pdf", "svg", "png"]:
            raise ValueError(f"Invalid format: {format}. Use 'pdf', 'svg', or 'png' instead.")
        if not filename.endswith("." + format):
            filename += "." + format
        plt.savefig(os.path.join(FIGURE_PATH, filename))

    def annotate_edges(self, annotations: list[str]):
        """Annotate the edges with the given annotations"""
        for edge_index, annotation in enumerate(annotations):
            start, end = self.edges[edge_index]
            start_pos = self.node_positions[start]
            end_pos = self.node_positions[end]
            # Add text annotation for the probability
            mid_pos = (start_pos + end_pos) / 2
            # Offset the text position slightly for opposing edges
            offset = 0.16  # Offset for the text annotations
            if start < end:
                text_pos = mid_pos + np.array([offset, offset])
            else:
                text_pos = mid_pos - np.array([offset, offset])
            self.ax.text(
                text_pos[0],
                text_pos[1],
                f"{annotation}",
                fontsize=9,
                ha="center",
                color="black",
                zorder=3,
            )

    def display_training_data_distribution(self, training_data: list[tuple[int, int]]):
        """Display the training data on the warehouse plot

        :param training_data: List of tuples containing the edge index and the successor edge index
        """
        training_data_distribution_edges = [len([sample for sample in training_data if sample[0] == i]) for i in range(len(self.edges))]

        training_data_distribution = []
        for i in range(len(self.nodes)):
            training_data_distribution.append(0)
            for j in range(len(self.edges)):
                if self.edges[j][1] == i:
                    training_data_distribution[-1] += training_data_distribution_edges[j]
                

        # Normalize the training data distribution for scatter plot sizes
        sizes = [50 * count for count in training_data_distribution]

        # Plot the scatter points at the tip of each edge
        for node, size in zip(self.nodes, sizes):
            plt.scatter(node["x"], node["y"], s=size, c="red", alpha=0.6, zorder = -1)

        plt.title("Training Data Distribution")
