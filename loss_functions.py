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

def assert_all_good(value):
    assert torch.isnan(value).sum().item() == 0
    assert torch.isinf(value).sum().item() == 0

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

        # Compute true energy
        energy = energy_model(embeddings, y)
        assert_all_good(energy)
        # Compute the true score
        true_score = score(energy, pred_probs)
        assert_all_good(true_score)
        # Get samples
        sample_scores = []
        for y_hat in sample(pred_probs, self.K):
            sample_energy = energy_model(embeddings, y_hat)
            assert_all_good(sample_energy)
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