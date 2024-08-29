import json
import numpy as np
import os
from .constants import GRAPH_PATH
import torch

class GNNTracker:
    def __init__(self, 
                 T_step, 
                 N_humans_max=10, 
                 train=False, 
                 get_observability=lambda: None,
                 get_belonging_node=lambda: None):
        """Tracker based on a graph neural network.
        """

        # save input arguments
        self.T_step = T_step
        self.N_humans_max = N_humans_max
        self.train = train
        self.get_observability = lambda: np.array((get_observability()))
        self.get_belonging_node = get_belonging_node()

        # load graph
        with open(GRAPH_PATH, 'r') as f:
            graph_data = json.load(f)
        self.nodes = graph_data['nodes']
        self.N_nodes = len(self.nodes)
        self.edges = graph_data['edges']

        # initialize variables
        self.P = np.zeros((self.N_nodes, self.N_humans_max))

    def add_observation(self, robot_perceptions):
        """Update P based on a set of robot perceptions.
        """
        for robot_perception in robot_perceptions:
            robot_position = robot_perception['ego_position']
            perceived_humans = robot_perception['perceived_humans']
            
            # calculate if a node is observable from the robot's position
            observability = self.get_observability(robot_position)

            # assign the observed humans to a node 
            human_node_ids = []
            for human in perceived_humans:
                human_position = human['pos_mean']
                human_node_ids.append(self.get_belonging_node(human_position))

            # update P based on the nodes that are observable but no human is 
            observability_without_human_nodes = observability
            observability_without_human_nodes[np.array(human_node_ids)] = 0
            self.P *= (1 - observability_without_human_nodes)
            self.P = self.P / np.sum(self.P, axis=0)

            # assign the perceived humans to the columns in P
            P_observed = np.zeros(self.P.shape)
            for human_node_id in human_node_ids:
                w = np.ones(self.N_nodes) / self.N_nodes
                P_observed[human_node_id] += w
            # TODO



    def predict(self):
        """Predict the next state of the P matrix.
        """

        # P' = f(P)

        return 1 - np.prod(1 - self.P, axis=1) # probability of at least one human at each node
