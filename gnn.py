from dgl.nn.pytorch.conv import GraphConv, GATConv
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


class GCNGlobalNorm(nn.Module):
    def __init__(self, in_feats, h_feats, mlp_dim, n_gcn_layers, n_mlp_layers, gcn_activation):
        super(GCNGlobalNorm, self).__init__()
        self.h_feats = h_feats
        self.projection = nn.Linear(in_feats, h_feats)

        self.node_layers = nn.ModuleList()
        self.graph_layers = nn.ModuleList()
        for i in range(n_gcn_layers):
            self.node_layers.append(GraphConv(h_feats, h_feats, activation=gcn_activation))
            self.graph_layers.append(nn.Linear(h_feats, h_feats))

        self.graph_pooling = dgltorch.glob.SumPooling()

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(h_feats, mlp_dim))
        for i in range(n_mlp_layers):
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Linear(mlp_dim, mlp_dim))

    def forward(self, g, node_features):
        h = self.projection(node_features)
        hg = self.graph_pooling(g, h)
        for i, layer in enumerate(self.node_layers):
            h = F.layer_norm(h + layer(g, h), (self.h_feats,))
            hg_i = self.graph_pooling(g, h)
            hg = hg + F.leaky_relu(self.graph_layers[i](hg_i))

        # Pooling
        for i, layer in enumerate(self.mlp):
            hg = layer(hg)

        return hg
    

class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, mlp_dim, n_gcn_layers, n_mlp_layers, gcn_activation):
        super(GAT, self).__init__()
        self.h_feats = h_feats
        self.projection = nn.Linear(in_feats, h_feats)

        self.node_layers = nn.ModuleList()
        self.graph_layers = nn.ModuleList()
        for i in range(n_gcn_layers - 1):
            self.node_layers.append(GATConv(h_feats, h_feats, num_heads=8, residual=True, activation=gcn_activation))
            self.graph_layers.append(nn.Linear(h_feats, h_feats))

        self.node_layers.append(GATConv(h_feats, h_feats, num_heads=1, residual=True, activation=gcn_activation))
        self.graph_layers.append(nn.Linear(h_feats, h_feats))

        self.graph_pooling = dgltorch.glob.SumPooling()

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(h_feats, mlp_dim))
        for i in range(n_mlp_layers):
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Linear(mlp_dim, mlp_dim))

    def forward(self, g, node_features):
        h = self.projection(node_features)
        hg = self.graph_pooling(g, h)
        assert len(self.node_layers) == len(self.graph_layers)
        for i in range(len(self.node_layers)):
            node_layer = self.node_layers[i]
            h = F.dropout(h)
            h = F.layer_norm(node_layer(g, h), (self.h_feats,))
            h = h.mean(1)
            graph_layer = self.graph_layers[i]
            hg_i = F.dropout(self.graph_pooling(g, h))
            hg = hg + F.leaky_relu(graph_layer(hg_i))

        '''
        for i, layer in enumerate(self.node_layers):
            h = F.dropout(h)
            h = F.layer_norm(h + layer(g, h), (self.h_feats,))
            hg_i = F.dropout(self.graph_pooling(g, h))
            hg = hg + F.leaky_relu(self.graph_layers[i](hg_i))
        '''
        # Pooling
        #hg = self.graph_pooling(g, h)
        #hg = F.softmax(self.graph_pooling(g, h))
        for i, layer in enumerate(self.mlp):
            hg = layer(hg)

        return hg