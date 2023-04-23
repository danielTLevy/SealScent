from dgl.nn.pytorch.conv import GraphConv
import dgl.nn.pytorch as dgltorch
import torch
from torch import nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, mlp_dim, n_gcn_layers, n_mlp_layers, gcn_activation):
        super(GCN, self).__init__()
        self.projection = nn.Linear(in_feats, h_feats)

        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(h_feats, h_feats, activation=gcn_activation))
        for i in range(n_gcn_layers - 1):
            self.layers.append(GraphConv(h_feats, h_feats, activation=gcn_activation))
        self.layers.append(GraphConv(h_feats, h_feats, activation=None))

        self.graph_pooling = dgltorch.glob.SumPooling()

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(h_feats, mlp_dim))
        for i in range(n_mlp_layers):
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Linear(mlp_dim, mlp_dim))

    def forward(self, g, node_features):
        h = self.projection(node_features)

        for i, layer in enumerate(self.layers):
            h = h + layer(g, h)
            h = F.relu(h)

        # Pooling
        hg = self.graph_pooling(g, h)
        #hg = F.softmax(self.graph_pooling(g, h))
        for i, layer in enumerate(self.mlp):
            hg = layer(hg)

        return hg
    
class GCN_direct(nn.Module):
    def __init__(self, in_feats, h_feats, mlp_dim, n_gcn_layers, n_mlp_layers, gcn_activation, num_classes):
        super(GCN_direct, self).__init__()
        self.projection = nn.Linear(in_feats, h_feats)

        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(h_feats, h_feats, activation=gcn_activation))
        for i in range(n_gcn_layers - 1):
            self.layers.append(GraphConv(h_feats, h_feats, activation=gcn_activation))
        self.layers.append(GraphConv(h_feats, h_feats, activation=None))

        self.graph_pooling = dgltorch.glob.SumPooling()

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(h_feats, mlp_dim))
        for i in range(n_mlp_layers - 1):
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Linear(mlp_dim, mlp_dim))
        self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(mlp_dim, num_classes))

    def forward(self, g, node_features):
        h = self.projection(node_features)

        for i, layer in enumerate(self.layers):
            h = h + layer(g, h)
            h = F.relu(h)

        # Pooling
        hg = self.graph_pooling(g, h)
        #hg = F.softmax(self.graph_pooling(g, h))
        for i, layer in enumerate(self.mlp):
            hg = layer(hg)

        return torch.sigmoid(hg)
    