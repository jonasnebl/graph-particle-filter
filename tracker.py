import json
import numpy as np

class Tracker:
    def __init__(self):
        with open('graph_data.json', 'r') as f:
            graph_data = json.load(f)
        self.nodes = graph_data['nodes']
        self.probabilites = np.random.uniform(0, 1, size=(len(self.nodes),))

    def add_observation(self, state):
        perceived_probabilities = []
        perceived_confidences = []
        for agent in state:
            if agent['type'] == 'robot':
                perceived_probabilities.append([node[0] for node in agent['perception_extended']])
                perceived_confidences.append([node[1] for node in agent['perception_extended']])

        for i, node in enumerate(self.nodes):
            # Update the probabilities
            self.probabilites[i] = np.sum(
                [perceived_probabilities[j][i] for j in range(len(perceived_probabilities))])
            if self.probabilites[i] > 1:
                self.probabilites[i] = 1



        print(perceived_probabilities)
        print(perceived_confidences)

    def predict(self):
        pass # all humans stay where they are
