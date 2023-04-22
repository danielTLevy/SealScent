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
import wandb
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

def setup_wandb(cfg):
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'project': cfg.wandb.project, 'entity': cfg.wandb.entity,
             'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True),
              'reinit': True,
              'mode': cfg.wandb.mode}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(np.array(labels))

def make_pred(graph, model):
    features = graph.ndata['h']
    pred, _ = model(graph, features)
    return pred

@hydra.main(config_path='configs/', config_name='config')
def main(cfg: DictConfig):
    cfg = setup_wandb(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    full_dataset = LeffingwellDataset()
    split_fracs = [cfg.dataset.frac_train, cfg.dataset.frac_val, 1 - cfg.dataset.frac_train - cfg.dataset.frac_val]
    train_data, val_data, test_data = dgl.data.utils.split_dataset(full_dataset, split_fracs)

    train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, shuffle=True,
                         collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=cfg.dataset.batch_size, shuffle=True,
                         collate_fn=collate)
    node_feat_length = full_dataset.NODE_FEAT_LENGTH
    n_labels = full_dataset.N_LABELS


    gcn = GCN(in_feats=node_feat_length, h_feats=cfg.model.h_feats, mlp_dim=cfg.model.hidden_dim, n_gcn_layers=3, n_mlp_layers=2, gcn_activation=F.leaky_relu)
    task_net = TaskNet(n_labels=n_labels, hidden_dim=cfg.model.hidden_dim, gnn=gcn)
    task_net = task_net.to(device)

    epoch = 0

    # Note: an untrained model will have a loss of about 0.69 on average
    # Trivial model of prediction all 0s will have a loss of 4.42 on average
    # Using a model seems to rapidly get to 0.12 loss on average

    optimizer = torch.optim.Adam(task_net.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    weighted_loss_fcn = nn.BCELoss(weight = torch.Tensor(full_dataset.label_weights).to(device), reduction='sum')
    unweighted_loss_fcn = nn.BCELoss(reduction='mean')
    if cfg.training.overfit == 0:
        n_train_graphs = len(train_loader)
    else:
        n_train_graphs = cfg.training.overfit
    epoch_iter = tqdm(range(cfg.training.epochs))


    for epoch_i in epoch_iter:
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
        for i, (graph, labels) in enumerate(train_loader):
            if cfg.training.overfit > 0 and i > cfg.training.overfit:
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

            if cfg.training.weighted_loss:
                loss  = weighted_bce
            else:
                loss = unweighted_bce
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
            for graph, labels in val_loader:
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

        wandb.log(epoch_stats)
        epoch_iter.set_postfix(**epoch_stats)
        epoch += 1

if __name__ == "__main__":
    main()