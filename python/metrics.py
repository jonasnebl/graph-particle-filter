import numpy as np
from KDEpy import FFTKDE
import matplotlib.pyplot as plt

class Confidence:
    def __init__(self, sim_log, node_probabilities):
        """Confidence metric. Rewards estimated probabilities close to 0.
        """
        self.sim_log = sim_log
        self.node_probabilities = np.array(node_probabilities)

    def per_node(self, only_count_nodes_with_no_humans=True):
        """Confidence evaluated for each graph node specifically
        """
        is_human_at_the_node = np.zeros((self.node_probabilities.shape), dtype=bool)
        if only_count_nodes_with_no_humans:
            for i, state in enumerate(self.sim_log['sim_states']):
                for agent in state:
                    if agent['type'] == 'human':
                        is_human_at_the_node[i, agent['belonging_node']] = True
        return np.mean(~is_human_at_the_node * (1 - np.exp(- self.node_probabilities)), axis=0)

    def per_graph(self, only_count_nodes_with_no_humans=True):
        """Confidence averaged over the whole graph
        """
        return self.per_node(only_count_nodes_with_no_humans).mean()


class Accuracy:
    def __init__(self, sim_log, node_probabilities):
        """Accuracy metric. Rewards correctly estimated probabilities."""
        self.sim_log = sim_log
        self.node_probabilities = np.array(node_probabilities)
        self.N_trapezoids = 100
        self.bandwidth = 0.1
        self.e = 1e-4

    def per_node(self):
        """Accuracy evaluated for each graph node specifically
        """
        is_human_at_the_node = np.zeros((self.node_probabilities.shape), dtype=bool)
        for i, state in enumerate(self.sim_log['sim_states']):
            for agent in state:
                if agent['type'] == 'human':
                    is_human_at_the_node[i, agent['belonging_node']] = True

        node_accuracies = []
        for node in range(self.node_probabilities.shape[1]):
            data_human_at_node = self.node_probabilities[is_human_at_the_node[:,node], node]
            data_no_human_at_node = self.node_probabilities[~is_human_at_the_node[:,node], node]

            # perform kernel density estimation
            N_p = data_human_at_node.size
            N_q = data_no_human_at_node.size
            if N_p != 0:
                p_kde = FFTKDE(kernel='gaussian', bw=self.bandwidth).fit(data_human_at_node)
            else: 
                p_kde = lambda x: np.ones(x.shape)
            if N_q != 0:
                q_kde = FFTKDE(kernel='gaussian', bw=self.bandwidth).fit(data_no_human_at_node)
            else:
                q_kde = lambda x: np.ones(x.shape)

            x = np.linspace(-self.e, 1+self.e, self.N_trapezoids)
            node_accuracies.append(np.trapz(
                (x - (N_p * p_kde(x))/(N_p * p_kde(x) + N_q * q_kde(x)))**2
                    * (N_p * p_kde(x) + N_q * q_kde(x)) / (N_p + N_q),
                x
            ))

        return np.array(node_accuracies)
    
    def per_graph(self):
        return self.per_node().mean()
