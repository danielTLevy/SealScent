import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eps = 1e-8
def score(energy, pred_probs):
    return -energy - torch.sum(torch.log(pred_probs + eps), dim=-1, keepdim=True)

def sample(probs, K):
    '''
    Sample K different sets of labels using the probability distribution
    output by the task model
    '''
    for k in range(K):
        y = torch.bernoulli(probs)
        yield y


class NCELoss(nn.Module):
    '''
    Loss for the energy net
    '''
    def __init__(self, K):
        super(NCELoss, self).__init__()
        self.K = K
            
    def forward(self, pred_probs, energy_model, y, embeddings=None, x_graph=None, x_feat=None):
        '''
        Compute the NCE loss
        '''
         # Compute true energy
        if embeddings is None:
            # Calculate embeddings
            embeddings = energy_model.get_embeddings(x_graph, x_feat)
        energy = energy_model.forward_embeddings(embeddings, y)
        
        # Compute the true score
        true_score = score(energy, pred_probs)
        # Get samples
        sample_scores = []
        for y_hat in sample(pred_probs, self.K):
            sample_energy = energy_model.forward_embeddings(embeddings, y_hat)
            sample_score = score(sample_energy, pred_probs)
            sample_scores.append(sample_score)
        sample_scores = torch.stack(sample_scores)

        return -true_score + torch.logsumexp(sample_scores, dim=0)

class SealTaskLoss(nn.Module):
    '''
    Loss for the task net
    '''
    def __init__(self, lam, weighted=True, label_weights=None):
        super(SealTaskLoss, self).__init__()
        self.lam = lam
        self.label_weights_tensor = torch.Tensor(label_weights).to(device)

        if weighted:
            self.ce_loss_fcn = self.weighted_bce
        else:
            self.ce_loss_fcn = self.unweighted_bce

    def unweighted_bce(self, pred, y):
        return self.cross_entropy(pred, y)
    
    def weighted_bce(self, pred, y):
        return self.cross_entropy(pred, y, weights=self.label_weights_tensor)

    def cross_entropy(self, pred, y, weights = None):
        losses = (y*torch.log(pred + eps) + (1-y)*torch.log(1-pred + eps))
        if weights is not None:
            losses = len(weights)*weights*losses
        return -torch.sum(losses, dim=-1, keepdim=True)

    def forward(self, pred, y, energy):
        '''
        Compute the SEAL loss
        '''
        # Compute the cross-entropy loss
        ce_loss = self.ce_loss_fcn(pred, y)

        # Compute the energy model loss
        return ce_loss + self.lam*energy