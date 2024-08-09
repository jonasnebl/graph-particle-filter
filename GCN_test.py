import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import json
import numpy as np

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

with open('graph_data.json', 'r') as f:
    graph_data = json.load(f)
nodes = graph_data['nodes']
N_nodes = len(nodes)
edges = graph_data['edges']
probabilites = 0.1 * np.ones((N_nodes,))      

data = Data(x=torch.tensor(probabilites[np.newaxis,:]),
            edge_index=torch.tensor(edges).T
            )
    
print(data)