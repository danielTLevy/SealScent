import torch
from torch import nn

def score(energy, pred_probs):
    return -energy - torch.sum(torch.log(pred_probs))

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
            
    def forward(self, pred_probs, embeddings, energy_model, y):
        '''
        Compute the NCE loss
        '''
        with torch.no_grad():
            pred_probs

        # Compute true energy
        energy = energy_model(embeddings, y)

        # Compute the true score
        true_score = score(energy, pred_probs)

        # Get samples
        sample_scores = []
        for y_hat in sample(pred_probs, self.K):
            sample_energy = energy_model(embeddings, y_hat)
            sample_scores.append(score(sample_energy, pred_probs))
        sample_scores = torch.stack(sample_scores)

        return -true_score + torch.logsumexp(sample_scores, dim=0)

class SealTaskLoss(nn.Module):
    '''
    Loss for the task net
    '''
    def __init__(self, lam, weighted=True, label_weights=None):
        super(SealTaskLoss, self).__init__()
        self.lam = lam
        if weighted:
            self.ce_loss_fcn = nn.BCELoss(weight = torch.Tensor(label_weights).to(device), reduction='sum')
        else:
            self.ce_loss_fcn = nn.BCELoss(reduction='sum')

    def forward(self, pred, y, energy):
        '''
        Compute the SEAL loss
        '''
        # Compute the cross-entropy loss
        ce_loss = self.ce_loss_fcn(pred, y)

        # Compute the energy model loss
        return ce_loss + self.lam*energy