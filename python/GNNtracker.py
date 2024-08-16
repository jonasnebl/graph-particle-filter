import json
import numpy as np
import os
from .constants import GRAPH_PATH
import torch

class GNNTracker:
    def __init__(self, T_step, train=False):
        with open(GRAPH_PATH, 'r') as f:
            graph_data = json.load(f)
        self.nodes = graph_data['nodes']
        self.N_nodes = len(self.nodes)
        self.edges = graph_data['edges']

        self.probabilites = 0.1 * np.ones((self.N_nodes,))   

        # model = torch.load('models/gcn_model_full.pth')
        # model.to('cpu')
        # model.eval()

    @staticmethod
    def extract_observation_from_state(state):
        """Extract the perceived probabilities and confidences from the enhanced robot perception provided by the C++ simulation and combine them for all robotss.
        """
        perceived_probabilities = []
        perceived_confidences = []
        for agent in state:
            if agent['type'] == 'robot':
                perceived_probabilities.append([node[0] for node in agent['perception_extended']])
                perceived_confidences.append([node[1] for node in agent['perception_extended']])

        # combine the perceived probabilities and confidences of all robots
        perceived_probabilities = np.clip(np.sum(perceived_probabilities, axis=0), 0, 1)
        perceived_confidences = np.clip(np.sum(perceived_confidences, axis=0), 0, 1)

        return perceived_probabilities, perceived_confidences

    def add_observation(self, state):
        perceived_probabilities, perceived_confidences = self.extract_observation_from_state(state)

        # Update the probabilities of the nodes
        self.probabilites = np.where(perceived_confidences == 1, perceived_probabilities, self.probabilites)

    def predict(self):
        # spread the probabilities of the nodes to the edges
        spread_factor = 0.005
        for edge in self.edges:
            self.probabilites[edge[1]] += spread_factor * self.probabilites[edge[0]]
            self.probabilites[edge[0]] *= 1 - spread_factor

        return self.probabilites

        # use the GCN to predict the probabilities

        # self.probabilites = self.model(torch.tensor(self.probabilites).unsqueeze(0).float()).detach().numpy()
