import torch
from torch import nn
import torch.nn.functional as F

class EnergyModel(nn.Module):
    def __init__(self, n_labels, global_dim, hidden_dim, lam=1):
        super(EnergyModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.global_dim = global_dim
        self.n_labels = n_labels
        self.lam = lam

        # Global portion
        self.global_proj = nn.Linear(n_labels, self.global_dim)
        self.global_func = nn.Linear(self.global_dim, 1)

        # Local portion
        self.label_proj = nn.Linear(self.hidden_dim, self.n_labels)

    
    def forward(self, embeddings, y):
        global_e = self.global_func(F.softplus(self.global_proj(y)))
        local_e = torch.sum(y * self.label_proj(embeddings))
        return global_e + self.lam*local_e

    #def forward(self, x_graph, x_feat, y):
    #    global_e = self.global_func(F.softplus(self.global_proj(y)))
    #    local_e = torch.sum(y * self.label_proj(self.gnn(x_graph, x_feat).squeeze()))
    #    return global_e + self.lam*local_e
    

