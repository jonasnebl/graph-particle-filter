import json
import numpy as np
import os
from .constants import GRAPH_PATH
import torch

class GNNTracker:
    def __init__(self, T_step, train=False, extend_perception=lambda: None):
        self.T_step = T_step
        self.train = train
        self.extend_perception = extend_perception

        with open(GRAPH_PATH, 'r') as f:
            graph_data = json.load(f)
        self.nodes = graph_data['nodes']
        self.N_nodes = len(self.nodes)
        self.edges = graph_data['edges']

        self.probabilites = 0.1 * np.ones((self.N_nodes,))   

    def add_observation(self, robot_perceptions):
        for robot_perception in robot_perceptions:
            robot_position = robot_perception['ego_position']
            perceived_humans = robot_perception['perceived_humans']
            extended_perceptions = self.extend_perception(
                robot_position, 
                [perceived_human["pos_mean"] for perceived_human in perceived_humans])        

    def predict(self):
        return self.probabilites
