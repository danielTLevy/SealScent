import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EnergyModel(nn.Module):
    def __init__(self, n_labels, global_dim, hidden_dim, gnn=None, lam=1):
        super(EnergyModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.global_dim = global_dim
        self.n_labels = n_labels
        self.lam = lam
        self.gnn = gnn

        # Global portion
        self.global_proj = nn.Linear(n_labels, self.global_dim)
        self.global_func = nn.Linear(self.global_dim, 1)

        # Local portion
        self.label_proj = nn.Linear(self.hidden_dim, self.n_labels)

    def get_embeddings(self, x_graph, x_feat):
        return self.gnn(x_graph, x_feat)

    def forward_embeddings(self, embeddings, y):
        global_e = self.global_func(F.softplus(self.global_proj(y)))
        local_e = torch.sum(y * self.label_proj(embeddings), dim=-1, keepdim=True)
        return global_e + self.lam*local_e

    def forward_graph(self, x_graph, x_feat, y):
        if self.gnn is None:
            raise NotImplementedError()
        embeddings = self.get_embeddings(x_graph, x_feat)
        global_e = self.global_func(F.softplus(self.global_proj(y)))
        local_e = torch.sum(y * self.label_proj(embeddings), dim=-1, keepdim=True)
        return global_e + self.lam*local_e



class DummyEnergy(nn.Module):
    def __init__(self):
        super(DummyEnergy, self).__init__()

    def get_embeddings(self, x_graph, x_feat):
        raise NotImplementedError()

    def forward_graph(self, x_graph, x_feat, y):
        return torch.zeros((y.shape[0], 1)).to(device)

    def forward_embeddings(self, embeddings, y):
        return torch.zeros((y.shape[0], 1)).to(device)

