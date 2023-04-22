import torch
import torch.nn as nn

class TaskNet(nn.Module):
    '''
    Class to project from GNN to labels
    '''
    def __init__(self, n_labels, hidden_dim, gnn):
        super(TaskNet, self).__init__()
        self.n_labels = n_labels
        self.hidden_dim = hidden_dim
        self.gnn = gnn
        self.label_proj = nn.Linear(self.hidden_dim, self.n_labels)

    def forward(self, x_graph, x_feat):
        embedding = self.gnn(x_graph, x_feat)
        label_preds = torch.sigmoid(self.label_proj(embedding))
        return label_preds, embedding
