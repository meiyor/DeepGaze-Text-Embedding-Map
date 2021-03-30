import numpy as np
#from pysaliency import roc
from pysaliency.roc import general_roc
import torch
#from torch import std_mean

def log_likelihood(log_density, fixation_mask, weights=None):
    #if weights is None:
    #    weights = torch.ones(log_density.shape[0])

    weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()

    dense_mask = fixation_mask.to_dense()
    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)
    ll = torch.mean(
        weights.type('torch.cuda.FloatTensor') * torch.sum(log_density.type('torch.cuda.FloatTensor') * dense_mask.type('torch.cuda.FloatTensor'), dim=(-1, -2), keepdim=True) / fixation_count.type('torch.cuda.FloatTensor')
    )
    return (ll + np.log(log_density.shape[-1] * log_density.shape[-2])) / np.log(2)

def log_likelihood_data(log_density, fixation_mask, weights=None):
    #if weights is None:
    #    weights = torch.ones(log_density.shape[0])

    weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()

    dense_mask = fixation_mask.to_dense()
    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)
    ll = weights.type('torch.cuda.FloatTensor') * torch.sum(log_density.type('torch.cuda.FloatTensor') * dense_mask.type('torch.cuda.FloatTensor'), dim=(-1, -2), keepdim=True) / fixation_count.type('torch.cuda.FloatTensor')
    return (ll + np.log(log_density.shape[-1] * log_density.shape[-2])) / np.log(2)


def log_likelihood_std(log_density, fixation_mask, weights=None):
    #if weights is None:
    #    weights = torch.ones(log_density.shape[0])

    weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()

    dense_mask = fixation_mask.to_dense()
    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)
    ll = torch.std(
        weights.type('torch.cuda.FloatTensor') * torch.sum(log_density.type('torch.cuda.FloatTensor') * dense_mask.type('torch.cuda.FloatTensor'), dim=(-1, -2), keepdim=True) / fixation_count.type('torch.cuda.FloatTensor')
    )
    return (ll + np.log(log_density.shape[-1] * log_density.shape[-2])) / np.log(2)

def nss(log_density, fixation_mask, weights=None):
    weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()
    dense_mask = fixation_mask.to_dense()
    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)

    density = torch.exp(log_density)
    mean = torch.mean(density, dim=(-1, -2), keepdim=True)
    std = torch.std(density, dim=(-1, -2), keepdim=True)
    saliency_map = (density - mean) / std

    nss = torch.mean(
        weights.type('torch.cuda.FloatTensor') * torch.sum(saliency_map.type('torch.cuda.FloatTensor') * dense_mask.type('torch.cuda.FloatTensor'), dim=(-1, -2), keepdim=True) / fixation_count.type('torch.cuda.FloatTensor')
    )
    return nss

def nss_data(log_density, fixation_mask, weights=None):
    weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()
    dense_mask = fixation_mask.to_dense()
    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)

    density = torch.exp(log_density)
    mean = torch.mean(density, dim=(-1, -2), keepdim=True)
    std = torch.std(density, dim=(-1, -2), keepdim=True)
    saliency_map = (density - mean) / std

    nss=weights.type('torch.cuda.FloatTensor') * torch.sum(saliency_map.type('torch.cuda.FloatTensor') * dense_mask.type('torch.cuda.FloatTensor'), dim=(-1, -2), keepdim=True) / fixation_count.type('torch.cuda.FloatTensor')
    return nss


def nss_std(log_density, fixation_mask, weights=None):
    weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()
    dense_mask = fixation_mask.to_dense()
    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)

    density = torch.exp(log_density)
    mean = torch.mean(density, dim=(-1, -2), keepdim=True)
    std = torch.std(density, dim=(-1, -2), keepdim=True)
    saliency_map = (density - mean) / std

    nss = torch.std(
        weights.type('torch.cuda.FloatTensor') * torch.sum(saliency_map.type('torch.cuda.FloatTensor') * dense_mask.type('torch.cuda.FloatTensor'), dim=(-1, -2), keepdim=True) / fixation_count.type('torch.cuda.FloatTensor')
    )
    return nss

def auc(log_density, fixation_mask, weights=None):
    weights = len(weights) * weights / weights.sum()
    # TODO: This doesn't account for multiple fixations in the same location!
    def image_auc(log_density, fixation_mask):
        dense_mask = fixation_mask#.to_dense()
        positives = torch.masked_select(log_density, dense_mask.type(torch.cuda.ByteTensor)).detach().cpu().numpy()
        negatives = log_density.flatten().detach().cpu().numpy()

        auc, _, _ = general_roc(positives.astype('double'), negatives.astype('double'))
        print('AUC:',auc)
        return torch.tensor(auc)#.type('torch.cuda.FloatTensor')
    fix_n=fixation_mask.to_dense()
    return torch.mean(weights.cpu() * torch.tensor([
        image_auc(log_density[i], fix_n[i]) for i in range(log_density.shape[0])
    ]).type('torch.DoubleTensor'))

def auc_data(log_density, fixation_mask, weights=None):
    weights = len(weights) * weights / weights.sum()
    # TODO: This doesn't account for multiple fixations in the same location!
    def image_auc(log_density, fixation_mask):
        dense_mask = fixation_mask#.to_dense()
        positives = torch.masked_select(log_density, dense_mask.type(torch.cuda.ByteTensor)).detach().cpu().numpy()
        negatives = log_density.flatten().detach().cpu().numpy()

        auc, _, _ = general_roc(positives.astype('double'), negatives.astype('double'))
        print('AUC:',auc)
        return torch.tensor(auc)#.type('torch.cuda.FloatTensor')
    fix_n=fixation_mask.to_dense()
    return weights.cpu() * torch.tensor([image_auc(log_density[i], fix_n[i]) for i in range(log_density.shape[0])]).type('torch.DoubleTensor')

def auc_std(log_density, fixation_mask, weights=None):
    weights = len(weights) * weights / weights.sum()
    # TODO: This doesn't account for multiple fixations in the same location!
    def image_auc(log_density, fixation_mask):
        dense_mask = fixation_mask#.to_dense()
        positives = torch.masked_select(log_density, dense_mask.type(torch.cuda.ByteTensor)).detach().cpu().numpy()
        negatives = log_density.flatten().detach().cpu().numpy()

        auc, _, _ = general_roc(positives.astype('double'), negatives.astype('double'))
        return torch.tensor(auc)#.type('torch.cuda.FloatTensor')
    fix_n=fixation_mask.to_dense()
    return torch.std(weights.cpu() * torch.tensor([
        image_auc(log_density[i], fix_n[i]) for i in range(log_density.shape[0])
    ]).type('torch.DoubleTensor'))
