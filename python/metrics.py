import numpy as np
from KDEpy import FFTKDE
import matplotlib.pyplot as plt


def calc_human_at_the_edge(sim_log, N_edges):
    is_human_at_the_edge = np.zeros((len(sim_log["sim_states"]), N_edges), dtype=bool)
    for i, state in enumerate(sim_log["sim_states"]):
        for agent in state:
            if agent["type"] == "human":
                is_human_at_the_edge[i, agent["belonging_edge"]] = True
    return is_human_at_the_edge


class Confidence:
    def __init__(self, sim_log, edge_probabilities):
        """Confidence metric. Rewards estimated probabilities close to 0."""
        self.sim_log = sim_log
        self.edge_probabilities = np.array(edge_probabilities)
        print(self.edge_probabilities.shape)

    def per_edge(self, only_count_nodes_with_no_humans=True):
        """Confidence evaluated for each graph node specifically"""
        if only_count_nodes_with_no_humans:
            is_human_at_the_node = calc_human_at_the_edge(self.sim_log, self.edge_probabilities.shape[1])
        else:
            is_human_at_the_node = np.zeros((self.edge_probabilities.shape), dtype=bool)
        return np.mean(~is_human_at_the_node * (1 - np.exp(-self.edge_probabilities)), axis=0)

    def per_graph(self, only_count_nodes_with_no_humans=True):
        """Confidence averaged over the whole graph"""
        return self.per_edge(only_count_nodes_with_no_humans).mean()


class Accuracy:
    def __init__(self, sim_log, edge_probabilities, bandwidth=1e-3):
        """Accuracy metric. Rewards correctly estimated probabilities."""
        self.sim_log = sim_log
        self.edge_probabilities = np.array(edge_probabilities)
        self.N_trapezoids = 1000
        self.bandwidth = bandwidth
        self.e = 1e-5

    def per_edge(self):
        """Accuracy evaluated for each graph node specifically"""
        is_human_at_the_node = calc_human_at_the_edge(self.sim_log, self.edge_probabilities.shape[1])

        node_accuracies = []
        for node in range(self.edge_probabilities.shape[1]):
            data_human_at_node = self.edge_probabilities[is_human_at_the_node[:, node], node]
            data_no_human_at_node = self.edge_probabilities[~is_human_at_the_node[:, node], node]

            # perform kernel density estimation
            N_p = data_human_at_node.size
            N_q = data_no_human_at_node.size
            if N_p != 0:
                p_kde = FFTKDE(kernel="gaussian", bw=self.bandwidth).fit(data_human_at_node)
            else:
                p_kde = lambda x: np.ones(x.shape)
            if N_q != 0:
                q_kde = FFTKDE(kernel="gaussian", bw=self.bandwidth).fit(data_no_human_at_node)
            else:
                q_kde = lambda x: np.ones(x.shape)

            x = np.linspace(-self.e, 1 + self.e, self.N_trapezoids)
            node_accuracies.append(
                np.trapz(
                    (x - (N_p * p_kde(x)) / (N_p * p_kde(x) + N_q * q_kde(x))) ** 2
                    * (N_p * p_kde(x) + N_q * q_kde(x))
                    / (N_p + N_q),
                    x,
                )
            )

        return np.array(node_accuracies)

    def per_graph(self):
        return self.per_edge().mean()


class MeanAveragePrecision:
    def __init__(self, sim_log, edge_probabilities):
        self.sim_log = sim_log
        self.edge_probabilities = np.array(edge_probabilities)
        self.N_thresholds = 100

    def per_edge(self):
        ground_truth_humans = calc_human_at_the_edge(self.sim_log, self.edge_probabilities.shape[1])
        thresholds = np.linspace(0, 1, self.N_thresholds)
        self.mean_average_precision = np.zeros((self.edge_probabilities.shape[1]))
        previous_recall = np.zeros((self.edge_probabilities.shape[1]))
        for threshold in thresholds:
            predicted_humans = self.edge_probabilities > threshold
            true_positive = np.sum(np.logical_and(predicted_humans, ground_truth_humans), axis=0)
            false_positive = np.sum(np.logical_and(predicted_humans, ground_truth_humans), axis=0)
            false_negative = np.sum(np.logical_and(~predicted_humans, ground_truth_humans), axis=0)
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            self.mean_average_precision += precision * (recall - previous_recall)
            previous_recall = recall

        return self.mean_average_precision

    def per_graph(self):
        per_edge_map = self.per_edge()
        return np.mean(per_edge_map[~np.isnan(per_edge_map)])
