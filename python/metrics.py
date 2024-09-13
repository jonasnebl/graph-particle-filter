import numpy as np
from KDEpy import FFTKDE
import matplotlib.pyplot as plt


def calc_human_at_the_node(sim_log, N_nodes):
    is_human_at_the_node = np.zeros((len(sim_log["sim_states"]), N_nodes), dtype=bool)
    for i, state in enumerate(sim_log["sim_states"]):
        for agent in state:
            if agent["type"] == "human":
                is_human_at_the_node[i, agent["belonging_node"]] = True
    return is_human_at_the_node


class Confidence:
    def __init__(self, sim_log, node_probabilities):
        """Confidence metric. Rewards estimated probabilities close to 0."""
        self.sim_log = sim_log
        self.node_probabilities = np.array(node_probabilities)

    def per_node(self, only_count_nodes_with_no_humans=True):
        """Confidence evaluated for each graph node specifically"""
        if only_count_nodes_with_no_humans:
            is_human_at_the_node = calc_human_at_the_node(self.sim_log, self.node_probabilities.shape[1])
        else:
            is_human_at_the_node = np.zeros((self.node_probabilities.shape), dtype=bool)
        return np.mean(~is_human_at_the_node * (1 - np.exp(-self.node_probabilities)), axis=0)

    def per_graph(self, only_count_nodes_with_no_humans=True):
        """Confidence averaged over the whole graph"""
        return self.per_node(only_count_nodes_with_no_humans).mean()


class Accuracy:
    def __init__(self, sim_log, node_probabilities, bandwidth=1e-3):
        """Accuracy metric. Rewards correctly estimated probabilities."""
        self.sim_log = sim_log
        self.node_probabilities = np.array(node_probabilities)
        self.N_trapezoids = 1000
        self.bandwidth = bandwidth
        self.e = 1e-5

    def per_node(self):
        """Accuracy evaluated for each graph node specifically"""
        is_human_at_the_node = calc_human_at_the_node(self.sim_log, self.node_probabilities.shape[1])

        node_accuracies = []
        for node in range(self.node_probabilities.shape[1]):
            data_human_at_node = self.node_probabilities[is_human_at_the_node[:, node], node]
            data_no_human_at_node = self.node_probabilities[~is_human_at_the_node[:, node], node]

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
        return self.per_node().mean()


class MeanAveragePrecision:
    def __init__(self, sim_log, node_probabilities):
        self.sim_log = sim_log
        self.node_probabilities = np.array(node_probabilities)
        self.N_thresholds = 100

    def per_node(self):
        ground_truth_humans = calc_human_at_the_node(self.sim_log, self.node_probabilities.shape[1])
        thresholds = np.linspace(0, 1, self.N_thresholds)
        self.mean_average_precision = np.zeros((self.node_probabilities.shape[1]))
        previous_recall = np.zeros((self.node_probabilities.shape[1]))
        for threshold in thresholds:
            predicted_humans = self.node_probabilities > threshold
            true_positive = np.sum(np.logical_and(predicted_humans, ground_truth_humans), axis=0)
            false_positive = np.sum(np.logical_and(predicted_humans, ground_truth_humans), axis=0)
            false_negative = np.sum(np.logical_and(~predicted_humans, ground_truth_humans), axis=0)
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            self.mean_average_precision += precision * (recall - previous_recall)
            previous_recall = recall

        return self.mean_average_precision

    def per_graph(self):
        per_node_map = self.per_node()
        return np.mean(per_node_map[~np.isnan(per_node_map)])
