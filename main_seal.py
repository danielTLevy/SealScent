import torch
from dataset import LeffingwellDataset
from energy_net import EnergyModel, DummyEnergy
from task_net import TaskNet
from loss_functions import NCELoss, SealTaskLoss
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



    gcn = GCN(in_feats=full_dataset.NODE_FEAT_LENGTH, h_feats=cfg.model.h_feats, mlp_dim=cfg.model.hidden_dim, n_gcn_layers=3, n_mlp_layers=2, gcn_activation=F.leaky_relu)
    task_net = TaskNet(n_labels=full_dataset.N_LABELS, hidden_dim=cfg.model.hidden_dim, gnn=gcn)
    task_net = task_net.to(device)
    optimizer_task = torch.optim.Adam(task_net.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    if cfg.model.use_energy:
        energy_net = EnergyModel(n_labels=full_dataset.N_LABELS, global_dim=cfg.model.energy_global_dim, hidden_dim=cfg.model.hidden_dim, lam=cfg.model.energy_global_local)
        energy_net = energy_net.to(device)
        optimizer_energy = torch.optim.Adam(energy_net.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    else:
        energy_net = DummyEnergy()
        cfg.model.lam = 0
        optimizer_energy = None

    energy_loss_fcn = NCELoss(K=cfg.model.K)
    task_loss_fcn = SealTaskLoss(lam=cfg.model.lam, weighted=True, label_weights=full_dataset.label_weights)

    if cfg.training.overfit == 0:
        n_train_graphs = len(train_loader)
    else:
        n_train_graphs = cfg.training.overfit
    epoch_iter = tqdm(range(cfg.training.epochs))


    for epoch_i in epoch_iter:
        all_train_preds = []
        all_train_labels = []

        epoch_stats = {
            'train_unweighted_bce': 0,
            'train_weighted_bce': 0,
            'train_macro_auroc': 0,
            'train_micro_auroc': 0,
            'val_macro_auroc': 0,
            'val_micro_auroc': 0,
            'energy': 0,
            'epoch': epoch_i,
            'task_loss': 0,
            'energy_loss': 0
        }
        for i, (graph, labels) in enumerate(train_loader):
            if cfg.training.overfit > 0 and i > cfg.training.overfit:
                continue
            graph = graph.to(device)
            graph_feats = graph.ndata['h']
            labels =  labels.float().to(device)
            batch_size = graph.batch_size


            ############################
            # Update the task model
            optimizer_task.zero_grad()
            energy_net.eval()
            # First, turn off all energy net gradients
            for param in energy_net.parameters():
                param.requires_grad = False
            # Next, turn on all task net gradients
            task_net.train()
            for param in task_net.parameters():
                param.requires_grad = True
            # Get predictions and embeddings
            pred, embeddings = task_net(graph, graph_feats)
            # Compute energy
            energy = energy_net(embeddings, labels)
            epoch_stats['energy'] += energy.sum().item()
            # Compute task loss
            task_net_loss = task_loss_fcn(pred, labels, energy).mean()
            task_net_loss.backward()
            epoch_stats['task_loss'] += task_net_loss.item()*batch_size
            optimizer_task.step()
            # Compute weighted and unweighted BCE
            unweighted_bce = task_loss_fcn.unweighted_bce(pred, labels).sum()
            weighted_bce = task_loss_fcn.weighted_bce(pred, labels).sum()
            epoch_stats['train_unweighted_bce'] += unweighted_bce.item()
            epoch_stats['train_weighted_bce'] += weighted_bce.item()

            all_train_preds.append(pred.detach().cpu().numpy())
            all_train_labels.append(labels.long().detach().cpu().numpy())

            ############################
            # Update the energy model
            if cfg.model.use_energy:
                optimizer_task.zero_grad()
                optimizer_energy.zero_grad()
                pred = pred.detach()
                embeddings = embeddings.detach()
                # Turn on all energy net gradients
                for param in energy_net.parameters():
                    param.requires_grad = True
                # Turn off all task net gradients
                for param in task_net.parameters():
                    param.requires_grad = False
                task_net.eval()
                energy_net.train()
                # Compute energy loss
                energy_net_loss = energy_loss_fcn(pred, embeddings, energy_net, labels).mean()
                energy_net_loss.backward()
                optimizer_energy.step()
                epoch_stats['energy_loss'] += energy_net_loss.item()*batch_size
                    


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
                all_val_labels.append(labels.long().numpy())

        # Normalize epoch stats
        epoch_stats['energy'] = epoch_stats['energy'] / n_train_graphs
        epoch_stats['energy_loss'] = epoch_stats['energy_loss'] / n_train_graphs
        epoch_stats['task_loss'] = epoch_stats['task_loss'] / n_train_graphs
        epoch_stats['train_weighted_bce'] =  epoch_stats['train_weighted_bce'] / n_train_graphs
        epoch_stats['train_unweighted_bce'] = epoch_stats['train_unweighted_bce'] / n_train_graphs
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

if __name__ == "__main__":
    main()