import json
import numpy as np

class Tracker:
    def __init__(self):
        with open('graph_data.json', 'r') as f:
            graph_data = json.load(f)
        self.nodes = graph_data['nodes']
        self.N_nodes = len(self.nodes)
        self.edges = graph_data['edges']
        self.probabilites = 0.1 * np.ones((self.N_nodes,))

    def add_observation(self, state):
        # extract the perceived probabilities and confidences from the enhanced robot perception provided by the C++ simulation
        perceived_probabilities = []
        perceived_confidences = []
        for agent in state:
            if agent['type'] == 'robot':
                perceived_probabilities.append([node[0] for node in agent['perception_extended']])
                perceived_confidences.append([node[1] for node in agent['perception_extended']])

        # combine the perceived probabilities and confidences of all robots
        perceived_probabilities = np.clip(np.sum(perceived_probabilities, axis=0), 0, 1)
        perceived_confidences = np.clip(np.sum(perceived_confidences, axis=0), 0, 1)

        # Update the probabilities of the nodes
        self.probabilites = np.where(perceived_confidences == 1, perceived_probabilities, self.probabilites)

    def predict(self):
        # spread the probabilities of the nodes to the edges
        spread_factor = 0.005
        for edge in self.edges:
            self.probabilites[edge[1]] += spread_factor * self.probabilites[edge[0]]
            self.probabilites[edge[0]] *= 1 - spread_factor
