import torch
from dataset import LeffingwellDataset
from energy_net import EnergyModel
from task_net import TaskNet
from gnn import GCN
import torchmetrics
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import dgl

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frac_train = 0.7
    frac_val = 0.1

    full_dataset = LeffingwellDataset()
    split_fracs = [frac_train, frac_val, 1 - frac_train - frac_val]
    train_data, val_data, test_data = dgl.data.utils.split_dataset(full_dataset, split_fracs)
    node_feat_length = full_dataset.NODE_FEAT_LENGTH
    n_labels = full_dataset.N_LABELS


    def make_pred(graph, model):
        features = graph.ndata['h']
        pred, _ = model(graph, features)
        return pred


    hidden_dim = 100
    #gcn_direct = GCN_direct(in_feats=node_feat_length, h_feats=32, mlp_dim=hidden_dim, n_gcn_layers=3, n_mlp_layers=2, gcn_activation=F.leaky_relu, num_classes=n_labels)
    #gcn_direct = gcn_direct.to(device)
    gcn = GCN(in_feats=node_feat_length, h_feats=32, mlp_dim=hidden_dim, n_gcn_layers=3, n_mlp_layers=2, gcn_activation=F.leaky_relu)
    task_net = TaskNet(n_labels=n_labels, hidden_dim=hidden_dim, gnn=gcn)
    task_net = task_net.to(device)

    #make_pred(example_graph, gcn_direct)
    epoch = 0
    statistics = {
        'train_unweighted_bce': [],
        'train_weighted_bce': [],
        'train_macro_auroc': [],
        'train_micro_auroc': [],
        'val_macro_auroc': [],
        'val_micro_auroc': [],
        'epoch': []
    }

    # Note: an untrained model will have a loss of about 0.69 on average
    # Trivial model of prediction all 0s will have a loss of 4.42 on average
    # Using a model seems to rapidly get to 0.12 loss on average
    LR = 0.0001

    optimizer = torch.optim.Adam(task_net.parameters(), lr=LR, weight_decay=5e-4)
    n_epochs = 1000

    weighted_loss_fcn = nn.BCELoss(weight = torch.Tensor(full_dataset.label_weights).to(device), reduction='sum')
    unweighted_loss_fcn = nn.BCELoss(reduction='mean')
    overfit = 100000
    n_train_graphs = min(len(train_data), overfit)
    epoch_iter = tqdm(range(n_epochs))


    for epoch_i in epoch_iter:
        #print("Epoch {}".format(epoch))
        #dataset_iter = tqdm(dataset[:overfit])
        unweighted_bce_sum = 0
        weighted_bce_sum = 0
        all_train_preds = []
        all_train_labels = []

        epoch_stats = {
            'train_unweighted_bce': 0,
            'train_weighted_bce': 0,
            'train_macro_auroc': 0,
            'train_micro_auroc': 0,
            'val_macro_auroc': 0,
            'val_micro_auroc': 0,
            'epoch': epoch
        }
        for i, (cid, graph, labels) in enumerate(train_data):
            if i > overfit:
                continue
            optimizer.zero_grad()
            task_net.train()
            graph = graph.to(device)
            pred = make_pred(graph, task_net)
            
            labels_tensor =  torch.tensor(labels).float().to(device)
            unweighted_bce = unweighted_loss_fcn(pred,labels_tensor)
            unweighted_bce_sum += unweighted_bce.item()
            weighted_bce = weighted_loss_fcn(pred, labels_tensor)
            weighted_bce_sum += weighted_bce.item()

            loss  = weighted_bce
            loss.backward()
            #print(task_net.gnn.mlp[0].weight.grad.norm())

            optimizer.step()
            
            #dataset_iter.set_postfix(loss=loss.item())
            all_train_preds.append(pred.detach().cpu().numpy())
            all_train_labels.append(labels)

        # validation loop
        with torch.no_grad():
            task_net.eval()
            all_val_preds = []
            all_val_labels = []
            for cid, graph, labels in val_data:
                graph = graph.to(device)
                pred = make_pred(graph, task_net)            
                #dataset_iter.set_postfix(loss=loss.item())
                all_val_preds.append(pred.detach().cpu().numpy())
                all_val_labels.append(labels)

        epoch_stats['train_weighted_bce'] = weighted_bce_sum / n_train_graphs
        epoch_stats['train_unweighted_bce'] = unweighted_bce_sum / n_train_graphs
        all_train_preds_tensor = torch.tensor(np.vstack(all_train_preds))
        all_train_labels_tensor = torch.tensor(np.vstack(all_train_labels))
        epoch_stats['train_macro_auroc']  = torchmetrics.functional.auroc(all_train_preds_tensor, all_train_labels_tensor, task='multilabel', average='macro', num_labels=113).item()
        epoch_stats['train_micro_auroc']  = torchmetrics.functional.auroc(all_train_preds_tensor, all_train_labels_tensor, task='multilabel', average='micro', num_labels=113).item()
        
        all_val_preds_tensor = torch.tensor(np.vstack(all_val_preds))
        all_val_labels_tensor = torch.tensor(np.vstack(all_val_labels))
        epoch_stats['val_macro_auroc']  = torchmetrics.functional.auroc(all_val_preds_tensor, all_val_labels_tensor, task='multilabel', average='macro', num_labels=113).item()
        epoch_stats['val_micro_auroc']  = torchmetrics.functional.auroc(all_val_preds_tensor, all_val_labels_tensor, task='multilabel', average='micro', num_labels=113).item()

        epoch_iter.set_postfix(**epoch_stats)
        for key, value in epoch_stats.items():
            statistics[key].append(value)
        epoch += 1

if __name__ == "__main__":
    main()