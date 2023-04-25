import torch
from dataset import LeffingwellDataset
from energy_net import EnergyModel, DummyEnergy, EnergyModelEmbeddings
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
    

    gnn_model = GCN
    gnn_params = {
        'in_feats': node_feat_length,
        'h_feats': cfg.model.h_feats,
        'mlp_dim': cfg.model.hidden_dim,
        'n_gcn_layers': 3,
        'n_mlp_layers': 2,
        'gcn_activation': F.leaky_relu
    }
    gnn = gnn_model(**gnn_params)
    task_net = TaskNet(n_labels=full_dataset.N_LABELS, hidden_dim=cfg.model.hidden_dim, gnn=gnn)
    task_net = task_net.to(device)
    optimizer_task = torch.optim.Adam(task_net.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    if cfg.model.use_energy:
        if cfg.model.shared_gnn:
            energy_net = EnergyModelEmbeddings(n_labels=full_dataset.N_LABELS, global_dim=cfg.model.energy_global_dim, hidden_dim=cfg.model.hidden_dim, lam=cfg.model.energy_global_local)
        else:
            energy_net_gnn = gnn_model(**gnn_params) 
            energy_net = EnergyModel(n_labels=full_dataset.N_LABELS, global_dim=cfg.model.energy_global_dim, hidden_dim=cfg.model.hidden_dim, gnn=energy_net_gnn, lam=cfg.model.energy_global_local)
        energy_net = energy_net.to(device)
        optimizer_energy = torch.optim.Adam(energy_net.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    else:
        energy_net = DummyEnergy()
        cfg.model.lam = 0
        optimizer_energy = None

    energy_loss_fcn = NCELoss(K=cfg.model.K)
    task_loss_fcn = SealTaskLoss(lam=cfg.model.lam, weighted=True, label_weights=full_dataset.label_weights)

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    epoch_iter = tqdm(range(cfg.training.epochs))


    for epoch_i in epoch_iter:
        all_train_preds = []
        all_train_labels = []

        epoch_stats = {
            'epoch': epoch_i,
            'train/unweighted_bce': 0,
            'train/weighted_bce': 0,
            'train/macro_auroc': 0,
            'train/micro_auroc': 0,
            'train/mean_energy': 0,
            'train/mean_pred_energy': 0,
            'train/task_loss': 0,
            'train/energy_loss': 0,
            'train/abs_energy_gap': 0,
            'val/macro_auroc': 0,
            'val/micro_auroc': 0,
            'val/energy_loss': 0,
            'val/mean_energy': 0,
            'val/mean_pred_energy': 0,
            'val/abs_energy_gap': 0,
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
            # Compute predicted energy and true energy
            if cfg.model.use_energy and  not cfg.model.shared_gnn:
                pred_energy = energy_net(graph, graph_feats, pred)
                true_energy = energy_net(graph, graph_feats, labels)
            else:
                pred_energy = energy_net(embeddings, pred)
                true_energy = energy_net(embeddings, labels)
            epoch_stats['train/mean_pred_energy'] += pred_energy.mean().item()
            epoch_stats['train/mean_energy'] += true_energy.mean().item()
            epoch_stats['train/abs_energy_gap'] += abs(epoch_stats['train/mean_energy'] - epoch_stats['train/mean_pred_energy'])
            # Compute task loss
            task_net_loss = task_loss_fcn(pred, labels, pred_energy).mean()
            task_net_loss.backward()
            epoch_stats['train/task_loss'] += task_net_loss.item()
            optimizer_task.step()
            # Compute weighted and unweighted BCE
            unweighted_bce = task_loss_fcn.unweighted_bce(pred, labels).mean()
            weighted_bce = task_loss_fcn.weighted_bce(pred, labels).mean()
            epoch_stats['train/unweighted_bce'] += unweighted_bce.item()
            epoch_stats['train/weighted_bce'] += weighted_bce.item()

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
                if cfg.model.shared_gnn:
                    # Reuse embeddings
                    energy_net_loss = energy_loss_fcn(pred, energy_net, labels, embeddings=embeddings).mean()
                else:
                    # Calculate them
                    energy_net_loss = energy_loss_fcn(pred, energy_net, labels, x_graph=graph, x_feat=graph_feats).mean()
                energy_net_loss.backward()
                optimizer_energy.step()
                epoch_stats['train/energy_loss'] += energy_net_loss.item()
                    


        # validation loop
        with torch.no_grad():
            task_net.eval()
            energy_net.eval()
            all_val_preds = []
            all_val_labels = []
            for graph, labels in val_loader:
                graph = graph.to(device)
                graph_feats = graph.ndata['h']
                labels = labels.float().to(device)
                pred, embeddings = task_net(graph, graph_feats)
                if cfg.model.use_energy and  not cfg.model.shared_gnn:
                    pred_energy = energy_net(graph, graph_feats, pred)
                    true_energy = energy_net(graph, graph_feats, labels)
                    energy_net_loss = energy_loss_fcn(pred, energy_net, labels, x_graph=graph, x_feat=graph_feats).mean()

                else:
                    pred_energy = energy_net(embeddings, pred)
                    true_energy = energy_net(embeddings, labels)
                    energy_net_loss = energy_loss_fcn(pred, energy_net, labels, embeddings=embeddings).mean()

                    
                epoch_stats['val/mean_pred_energy'] += pred_energy.mean().item()
                epoch_stats['val/mean_energy'] += true_energy.mean().item()
                epoch_stats['val/abs_energy_gap'] = abs(epoch_stats['val/mean_energy'] - epoch_stats['val/mean_pred_energy'])
                epoch_stats['val/energy_loss'] += energy_net_loss.item()

                #dataset_iter.set_postfix(loss=loss.item())
                all_val_preds.append(pred.detach().cpu().numpy())
                all_val_labels.append(labels.cpu().long().numpy())

        # Normalize epoch stats
        epoch_stats['train/mean_energy'] = epoch_stats['train/mean_energy'] / n_train_batches
        epoch_stats['train/mean_pred_energy'] = epoch_stats['train/mean_pred_energy'] / n_train_batches
        epoch_stats['train/abs_energy_gap'] = epoch_stats['train/abs_energy_gap'] / n_train_batches
        epoch_stats['train/energy_loss'] = epoch_stats['train/energy_loss'] / n_train_batches
        epoch_stats['train/task_loss'] = epoch_stats['train/task_loss'] / n_train_batches
        epoch_stats['train/weighted_bce'] =  epoch_stats['train/weighted_bce'] / n_train_batches
        epoch_stats['train/unweighted_bce'] = epoch_stats['train/unweighted_bce'] / n_train_batches
        all_train_preds_tensor = torch.tensor(np.vstack(all_train_preds))
        all_train_labels_tensor = torch.tensor(np.vstack(all_train_labels))
        epoch_stats['train/macro_auroc']  = torchmetrics.functional.auroc(all_train_preds_tensor, all_train_labels_tensor, task='multilabel', average='macro', num_labels=n_labels).item()
        epoch_stats['train/micro_auroc']  = torchmetrics.functional.auroc(all_train_preds_tensor, all_train_labels_tensor, task='multilabel', average='micro', num_labels=n_labels).item()
        
        all_val_preds_tensor = torch.tensor(np.vstack(all_val_preds))
        all_val_labels_tensor = torch.tensor(np.vstack(all_val_labels))
        epoch_stats['val/macro_auroc']  = torchmetrics.functional.auroc(all_val_preds_tensor, all_val_labels_tensor, task='multilabel', average='macro', num_labels=n_labels).item()
        epoch_stats['val/micro_auroc']  = torchmetrics.functional.auroc(all_val_preds_tensor, all_val_labels_tensor, task='multilabel', average='micro', num_labels=n_labels).item()
        epoch_stats['val/energy_loss'] = epoch_stats['val/energy_loss'] / n_val_batches
        epoch_stats['val/mean_energy'] = epoch_stats['val/mean_energy'] / n_val_batches

        wandb.log(epoch_stats)
        epoch_iter.set_postfix(**epoch_stats)

if __name__ == "__main__":
    main()