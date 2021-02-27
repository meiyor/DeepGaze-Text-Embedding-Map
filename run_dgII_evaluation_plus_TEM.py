## all packages added here ** please remember to add all of them before running the traning or testing
## to run roc.c frst cythonize the roc.pyc depending on your architecture using ipython or cython
import os
import argparse
import schema
import subprocess
import numpy as np
import pandas as pd
import math
import yaml
from pprint import pprint
from collections import Mapping
from copy import deepcopy
import glob
import pickle
from datetime import datetime
from contextlib import contextmanager
import importlib
from collections import defaultdict, OrderedDict
from natsort import natsorted
from glom import glom
from tqdm import tqdm
from boltons.fileutils import mkdir_p
from boltons.iterutils import windowed
from pysaliency.plotting import visualize_distribution
from pysaliency.filter_datasets import iterate_crossvalidation
import torch
import torch.nn as nn
import torch.nn.functional as F

## add the new DeepGaze_TEM object to run the new architecture including TEM, Feature extractor is the fixed VGG encoder
from deepgaze import DeepGaze_TEM, FeatureExtractor

# use Adabound to prevent easier overffiting, and have the same level of SGD learning and the same speed of Adam 
from adabound import AdaBound

from layers import LayerNorm, Conv2dMultiInput, Bias, LayerNormMultiInput
from boltons.iterutils import chunked

## export pysaliency codes
from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel
from pysaliency.precomputed_models import HDF5Model
from pysaliency import precomputed_models
from pysaliency import models
from pysaliency.datasets import create_subset, create_subset_TEM
from pysaliency.dataset_config import load_dataset_from_config
from pysaliency import external_datasets
import pysaliency

# import necessary modified data, metrics and vg objects for the new network
from data import ImageDataset, ImageDataset_TEM, FixationDataset, ImageDatasetSampler, FixationMaskTransform
from metrics import log_likelihood, nss, auc, log_likelihood_std, nss_std, auc_std
from boltons.cacheutils import cached, LRU

import sys

### set this flags if you have multiple GPU cores you want to use in particular
#torch.cuda.set_device('cuda:0')
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# coding: utf-8

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# coding: utf-8


if '__file__' in globals():
    # executing from file
    is_notebook = False
    print("Executing from script")
    import os
    root_directory = os.path.dirname(os.path.realpath(__file__))
    print(root_directory)
    print(os.getcwd())

    import sys
    sys.path.append('.')

else:
    # executing from notebook
    print("executing in ipython kernel")
    is_notebook = True
    import os
    root_directory = os.getcwd()
    print(root_directory)

    print(os.getcwd())


parser = argparse.ArgumentParser()
parser.add_argument('--training-part', default=None, type=str)
parser.add_argument('--crossval-fold-number', default=None, type=int)
parser.add_argument('--sub-experiment-no', default=None, type=int)
parser.add_argument('--sub-experiment', default=None, type=str)

if is_notebook:
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

print("Arguments", args)


if args.sub_experiment is not None:
    new_root = os.path.join(root_directory, args.sub_experiment)
    if not os.path.isdir(new_root):
        raise ValueError("Invalid sub experiment", args.sub_experiment)
    root_directory = new_root

if args.sub_experiment_no is not None:
    import glob
    
    sub_experiment_candidates = glob.glob(os.path.join(root_directory, f'experiment{args.sub_experiment_no:04d}*'))

    if not sub_experiment_candidates:
        raise ValueError("No subexperiment with number", args.sub_experiment_no)
    elif len(sub_experiment_candidates) > 1:
        raise ValueError("Too many candidates with number", args.sub_experiment_no)
    
    root_directory, = sub_experiment_candidates


def convert_to_update_schema(full_schema, create_schema=True):
    """Remove all defaults from schema to allow it to be used as update"""
    if isinstance(full_schema, schema.Schema):
        full_schema = full_schema.schema
    
    if isinstance(full_schema, dict):
        ## read the schema format
        new_schema = {}
        for key, item in full_schema.items():
            if not isinstance(key, schema.Optional):
                # make optional
                key = schema.Optional(key)
            else:
                # make sure to create new optional without default instead of removing default from existing
                key = schema.Optional(key.key)
            
            if isinstance(item, (schema.Schema, dict)):
                item = convert_to_update_schema(item, create_schema=False)
            
            new_schema[key] = item
        
        if create_schema:
            new_schema = schema.Schema(new_schema)
        
        return new_schema

    return full_schema


number = schema.Schema(schema.Or(int, float))

readout_network_spec = schema.Schema({
    'input_channels': schema.Or(int, [int]),
    'layers': [{
        'channels': int,
        'name': str,
        schema.Optional('activation_fn', default='relu'): schema.Or(None, str),
        schema.Optional('activation_scope', default=None): schema.Or(None, str),
        schema.Optional('bias', default=True): bool,
        schema.Optional('layer_norm', default=False): bool,
        schema.Optional('batch_norm', default=False): bool,
    }]
})

model_spec = schema.Schema({
    'downscale_factor': number,
    'readout_factor': int,
    'saliency_map_factor': int,
    'included_previous_fixations': [int],
    'include_previous_x': bool,
    'include_previous_y': bool,
    'included_durations': [int],
    schema.Optional('fixated_scopes', default=[]): [str],
    'features': { str: {
        'type': str,
        schema.Optional('params', default={}): dict,
        'used_features': [str],
    }},
    'scanpath_network': schema.Or(readout_network_spec, None),
    'saliency_network': readout_network_spec,
    'saliency_network_TEM':readout_network_spec,
    'fixation_selection_network': readout_network_spec,
    'fixation_selection_network_TEM': readout_network_spec,
    'conv_all_parameters': readout_network_spec,
    'conv_all_parameters_trans': readout_network_spec,
})

dataset_spec = schema.Schema(schema.Or(str, {
    schema.Optional('name'): str,
    schema.Optional('stimuli'): str,
    schema.Optional('fixations'): str,
    schema.Optional('centerbias'): str,
    schema.Optional('filters', default=[]): [dict]
}))

crossvalidation_spec = schema.Schema({
    'folds': int,
    'val_folds': int,
    'test_folds': int,
})

optimizer_spec = schema.Schema({
    'type': str,
    schema.Optional('params', default={}): dict,
})

lr_scheduler_spec = schema.Schema({
    'type': str,
    schema.Optional('params', default={}): dict,
})

evaluation_spec = schema.Schema({
    schema.Optional('compute_metrics', default={}): {
        schema.Optional('metrics', default=['IG', 'LL', 'AUC', 'NSS']): [schema.Or('IG', 'LL', 'AUC', 'NSS')],
        schema.Optional('datasets', default=['training', 'validation', 'test']): [schema.Or('training', 'validation', 'test')]
        },
    schema.Optional('compute_predictions', default={}):  schema.Or({}, {
        'datasets': [schema.Or('training', 'validation', 'test')]}),
})


cleanup_spec = schema.Schema({
    schema.Optional('cleanup_checkpoints', default=False): bool
})


default_optimizer = optimizer_spec.validate({
    'type': 'torch.optim.Adam',
    'params': {'lr': 0.01}
})

default_scheduler = lr_scheduler_spec.validate({
    'type': 'torch.optim.lr_scheduler.MultiStepScheduler',
    'params': {
        'milestones': [10, 20, 30, 40, 50, 60, 70, 80]
    }
})


training_part_spec = schema.Schema({
    'name': str,
    'train_dataset': dataset_spec,
    
    schema.Optional('optimizer'): convert_to_update_schema(optimizer_spec),
    schema.Optional('lr_scheduler'): convert_to_update_schema(lr_scheduler_spec),
    schema.Optional('minimal_learning_rate'): number,
    
    schema.Optional('iteration_element'): schema.Or('fixation', 'image'),
    schema.Optional('averaging_element'): schema.Or('fixation', 'image'),
    schema.Optional('model'): convert_to_update_schema(model_spec),
    schema.Optional('training_dataset_ratio_per_epoch'): float,
    schema.Optional('centerbias'): str,
    schema.Optional('val_dataset'): dataset_spec,
    schema.Optional('test_dataset'): dataset_spec,
    schema.Optional('crossvalidation'): crossvalidation_spec,
    schema.Optional('validation_metric', default='IG'): schema.Or('IG', 'LL', 'AUC', 'NSS'),
    schema.Optional('validation_metrics', default=['LL', 'IG', 'AUC', 'NSS']): [schema.Or('IG', 'LL', 'AUC', 'NSS')],
    schema.Optional('startwith', default=None): schema.Or(str, None),
    schema.Optional('evaluation', default=evaluation_spec.validate({})): evaluation_spec,
    schema.Optional('batch_size'): int,
    schema.Optional('cleanup', default=cleanup_spec.validate({})): cleanup_spec,
    schema.Optional('final_cleanup', default=cleanup_spec.validate({})): cleanup_spec,
})

config_schema = schema.Schema({
    'model': model_spec,
    'training': {
        schema.Optional('optimizer', default=default_optimizer): optimizer_spec,
        schema.Optional('lr_scheduler', default=default_scheduler): lr_scheduler_spec,
        schema.Optional('minimal_learning_rate', default=0.000001): number,
        schema.Optional('batch_size', default=2): int,
        schema.Optional('iteration_element', default='fixation'): schema.Or('fixation', 'image'),
        schema.Optional('averaging_element', default='fixation'): schema.Or('fixation', 'image'),
        schema.Optional('training_dataset_ratio_per_epoch', default=0.25): float,
        
        'parts': [training_part_spec],
    }
})


## setting_up the confing file
root_directory='experiments_root/' ## define the root directory a-priori
config = yaml.load(open('config_dg2_TEM.yaml'))

config = config_schema.validate(config)
print(yaml.safe_dump(config))
config_schema.validate(config);


def dict_merge(dct, merge_dct):
    """ Recursive dict merge. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    
    return dct


def reverse_dict_merge(dct, fallback_dct):
    """ Recursive dict merge. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in fallback_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(fallback_dct[k], Mapping)):
            reverse_dict_merge(dct[k], fallback_dct[k])
        elif k in dct:
            # don't fallback
            pass
        else:
            dct[k] = fallback_dct[k]
    
    return dct

bare_training_config = dict(config['training'])
del bare_training_config['parts']
bare_training_config['model'] = config['model']

for part in config['training']['parts']:
    reverse_dict_merge(part, deepcopy(bare_training_config))


config_schema.validate(config);

if is_notebook:
    get_ipython().run_line_magic('matplotlib', 'inline')

    import matplotlib.pyplot as plt
    from IPython.display import display
else:
    import matplotlib
    matplotlib.use('agg')
    
    import matplotlib.pyplot as plt



def build_readout_network_from_config(readout_config):
    layers = OrderedDict()
    input_channels = readout_config['input_channels']
    
    for k, layer_spec in enumerate(readout_config['layers']):
        if layer_spec['layer_norm']:
            if isinstance(input_channels, int):
                layers[f'layernorm{k}'] = LayerNorm(input_channels)
            else:
                layers[f'layernorm{k}'] = LayerNormMultiInput(input_channels)


        if isinstance(input_channels, int):
            if readout_config['layers'][0]['name']=='convtrans':
              layers[f'conv{k}'] = nn.ConvTranspose2d(input_channels, layer_spec['channels'], (1, 1), bias=False)
            else:
              layers[f'conv{k}'] = nn.Conv2d(input_channels, layer_spec['channels'], (1, 1), bias=False)
        else:
            layers[f'conv{k}'] = Conv2dMultiInput(input_channels, layer_spec['channels'], (1, 1), bias=False)
        input_channels = layer_spec['channels']
        
        #assert not layer_spec['batch_norm']
        
        if layer_spec['bias']:
            layers[f'bias{k}'] = Bias(input_channels)
        
        if layer_spec['activation_fn'] == 'relu':
            layers[f'relu{k}'] = nn.ReLU()
        elif layer_spec['activation_fn'] == 'softplus':
            layers[f'softplus{k}'] = nn.Softplus()
        elif layer_spec['activation_fn'] == 'celu':
            layers[f'softplus{k}'] = nn.CELU()
        elif layer_spec['activation_fn'] == 'gelu':
            layers[f'gelu{k}'] = GELU()
        elif layer_spec['activation_fn'] == 'selu':
            layers[f'selu{k}'] = torch.nn.SELU()
        elif layer_spec['activation_fn'] is None:
            pass
        else:
            raise ValueError(layer_spec['activation_fn'])
    
    return nn.Sequential(layers)


def import_class(name):
    module_name, class_name = name.rsplit('.', 1)
    #module_name='vgg'
    print(module_name,class_name)
    if module_name == 'deepgaze_pytorch.features.vgg':
      module_name='vgg'
    if module_name == 'torch.optim':
      module_name='adabound'
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_model(model_config):
    assert len(model_config['features']) == 1
    features_key, = list(model_config['features'].keys())
    features_config = model_config['features'][features_key]
    
    feature_class = import_class(features_config['type'])
    features = feature_class(**features_config['params'])
    
    feature_extractor = FeatureExtractor(features, features_config['used_features'])
    #print('extractor',feature_extractor)
    saliency_network = build_readout_network_from_config(model_config['saliency_network'])
    saliency_network_TEM = build_readout_network_from_config(model_config['saliency_network_TEM'])
    if model_config['scanpath_network'] is not None:
        scanpath_network = build_readout_network_from_config(model_config['scanpath_network'])
    else:
        scanpath_network = None
    fixation_selection_network = build_readout_network_from_config(model_config['fixation_selection_network'])
    fixation_selection_network_TEM = build_readout_network_from_config(model_config['fixation_selection_network_TEM'])
    conv_all_parameters = build_readout_network_from_config(model_config['conv_all_parameters'])
    conv_all_parameters_trans = build_readout_network_from_config(model_config['conv_all_parameters_trans'])
    
    ### new model definition
    model = DeepGaze_TEM(
        features=feature_extractor,
        saliency_network=saliency_network,
        saliency_network_TEM=saliency_network_TEM,
        scanpath_network=scanpath_network,
        fixation_selection_network=fixation_selection_network,
        fixation_selection_network_TEM=fixation_selection_network_TEM,
        conv_all_parameters = conv_all_parameters,
        conv_all_parameters_trans=conv_all_parameters_trans,
        downsample=model_config['downscale_factor'],
        readout_factor=model_config['readout_factor'],
        saliency_map_factor=model_config['saliency_map_factor'],
        included_fixations=model_config['included_previous_fixations'],
    )
    
    for scope in model_config['fixated_scopes']:
        for parameter_name, parameter in model.named_parameters():
            if parameter_name.startswith(scope):
                print("Fixating parameter", parameter_name)
                parameter.requires_grad = False
    
    print("Remaining training parameters")
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(parameter_name)
    
    return model


baseline_performance = cached(LRU(max_size=3))(lambda model, *args, **kwargs: model.information_gain(*args, **kwargs))

def eval_epoch(model, dataset, device, baseline_model, TEM_model, metrics=None, averaging_element='fixation',name='None'):
    print("Averaging element", averaging_element)
    model.eval()
    
    if metrics is None:
        metrics = ['LL', 'IG', 'NSS', 'AUC']
    
    metric_scores = {}
    metric_scores_std = {}
    metric_functions = {
        'LL': log_likelihood,
        'NSS': nss,
        'AUC': auc,
    }
    metric_functions_std = {
        'LL': log_likelihood_std,
        'NSS': nss_std,
        'AUC': auc_std,
    }
    batch_weights = []
    with torch.no_grad():
        pbar = tqdm(dataset)
        n=0
        mval=[]
        sval=[]
        for batch in pbar:
            image = batch['image'].to(device)
            TEM = batch['TEM'].to(device)
            #print(image.shape,TEM.shape,'shape_n')
            centerbias = batch['centerbias'].to(device)
            centerbias_TEM = batch['centerbias_TEM'].to(device)
            fixation_mask = batch['fixation_mask'].to(device)
            x_hist = batch.get('x_hist', torch.tensor([])).to(device)
            y_hist = batch.get('y_hist', torch.tensor([])).to(device)
            weights = batch['weight'].to(device)
            durations = batch.get('durations', torch.tensor([])).to(device)
            log_density = model(image, TEM, centerbias, centerbias_TEM, x_hist=x_hist, y_hist=y_hist, durations=durations)
            for metric_name, metric_fn in metric_functions.items():
                if metric_name not in metrics:
                    continue
                metric_scores.setdefault(metric_name, []).append(metric_fn(log_density, fixation_mask, weights=weights).detach().cpu().numpy())
            for metric_name_std, metric_fn_std in metric_functions_std.items():
                if metric_name_std not in metrics:
                    continue
                metric_scores_std.setdefault(metric_name_std, []).append(metric_fn_std(log_density, fixation_mask, weights=weights).detach().cpu().numpy())
            batch_weights.append(weights.detach().cpu().numpy().sum())
            
            for display_metric in ['LL', 'NSS', 'AUC']:
                if display_metric in metrics:
                    pbar.set_description('{} {:.05f}'.format(display_metric, np.average(metric_scores[display_metric], weights=batch_weights)))
                    break
    
    data = {metric_name: np.average(scores, weights=batch_weights) for metric_name, scores in metric_scores.items()}
    data_s= {metric_name_std: np.average(scores_std, weights=batch_weights) for metric_name_std, scores_std in metric_scores_std.items()}
    
    print(data,'val_mean')
    print(data_s,'val_std')
    if 'IG' in metrics:
        baseline_ll = baseline_performance(baseline_model, dataset.dataset.stimuli, dataset.dataset.fixations, verbose=True, average=averaging_element)
        data['IG'] = data['LL']-baseline_ll
 
    return data

def train_epoch(model, dataset, optimizer, device):
    model.train()
    losses = []
    batch_weights = []
    
    pbar = tqdm(dataset)
    mval=[]
    sval=[]
    for batch in pbar:
        optimizer.zero_grad()

        image = batch['image'].to(device)
        TEM = batch['TEM'].to(device)
        centerbias = batch['centerbias'].to(device)
        centerbias_TEM = batch['centerbias_TEM'].to(device)
        fixation_mask = batch['fixation_mask'].to(device)
        x_hist = batch.get('x_hist', torch.tensor([])).to(device)
        y_hist = batch.get('y_hist', torch.tensor([])).to(device)
        weights = batch['weight'].to(device)
        durations = batch.get('durations', torch.tensor([])).to(device)

        log_density = model(image, TEM, centerbias, centerbias_TEM,  x_hist=x_hist, y_hist=y_hist, durations=durations)

        loss = -log_likelihood(log_density, fixation_mask, weights=weights)
        losses.append(loss.detach().cpu().numpy())
        batch_weights.append(weights.detach().cpu().numpy().sum())
        pbar.set_description('{:.05f}'.format(np.average(losses, weights=batch_weights)))
        loss.backward()
        optimizer.step()
        del image, centerbias, fixation_mask, x_hist, y_hist, weights, durations, log_density

    #del pbar, batch
    return np.average(losses, weights=batch_weights)


def restore_from_checkpoint(model, optimizer, scheduler, path):
    print("Restoring from", path)
    data = torch.load(path)
    #print(data,'data')
    if 'optimizer' in data:
        # checkpoint contains training progress
        model.load_state_dict(data['model'])
        print(data)
        optimizer.load_state_dict(data['optimizer'])
        scheduler.load_state_dict(data['scheduler'])
        torch.set_rng_state(data['rng_state'])
        return data['step'], data['loss']
    else:
        # checkpoint contains just a model
        missing_keys, unexpected_keys = model.load_state_dict(data, strict=False)
        if missing_keys:
            print("WARNING! missing keys", missing_keys)
        if unexpected_keys:
            print("WARNING! Unexpected keys", unexpected_keys)


def save_training_state(model, optimizer, scheduler, step, loss, path):
    data = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng_state': torch.get_rng_state(),
        'step': step,
        'loss': loss,
    }
    
    torch.save(data, path)


def plot_scanpath(x_hist, y_hist, x, y, ax):
    for (x1, x2), (y1, y2) in zip(windowed(x_hist, 2), windowed(y_hist, 2)):
        if x1==x2 and y1==y2:
            continue
        ax.arrow(x1, y1, x2-x1, y2-y1, length_includes_head=True, head_length=20, head_width=20, color='red', zorder=10, linewidth=2)

    x1 = x_hist[-1]
    y1 = y_hist[-1]
    x2 = x
    y2 = y
    ax.arrow(x1, y1, x2-x1, y2-y1, length_includes_head=True, head_length=20, head_width=20, color='blue', linestyle=':', linewidth=2, zorder=10)


def visualize(model, vis_data_loader):
    model.eval()
    
    device = next(model.parameters()).device
    #device = 'cuda:1'
    print('dev',device)      
    batch = next(iter(vis_data_loader))
    
    #print(batch,'hi',vis_data_loader)
    #print(iter(vis_data_loader))

    image = batch['image'].to(device)
    TEM = batch['TEM'].to(device)
    centerbias = batch['centerbias'].to(device)
    centerbias_TEM = batch['centerbias_TEM'].to(device)
    fixation_mask = batch['fixation_mask'].to(device)
    x_hist = batch.get('x_hist', torch.tensor([])).to(device)
    y_hist = batch.get('y_hist', torch.tensor([])).to(device)
    durations = batch.get('durations', torch.tensor([])).to(device)
    log_density = model(image, TEM, centerbias, centerbias_TEM, x_hist=x_hist, y_hist=y_hist, durations=durations)

    log_density = log_density.detach().cpu().numpy()
    fixation_indices = fixation_mask.coalesce().indices().detach().cpu().numpy()
    rgb_image = image.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
    x_hist = x_hist.detach().cpu().numpy()
    y_hist = y_hist.detach().cpu().numpy()
    
    width = 4.0
    height = width / rgb_image.shape[2] * rgb_image.shape[1]
    f, axs = plt.subplots(len(rgb_image), 2, figsize=(2*width, height*len(rgb_image)))
    
    for row in range(len(rgb_image)):
        axs[row, 0].imshow(rgb_image[row])

        bs, ys, xs = fixation_indices

        ys = ys[bs == row]
        xs = xs[bs == row]

        if len(x_hist):
            _x_hist = x_hist[row]
        else:
            _x_hist = []

        if len(y_hist):
            _y_hist = y_hist[row]
        else:
            _y_hist = []

        visualize_distribution(log_density[row], ax=axs[row, 1])

        if len(_x_hist):
            plot_scanpath(_x_hist, _y_hist, xs[0], ys[0], axs[row, 0])
            plot_scanpath(_x_hist, _y_hist, xs[0], ys[0], axs[row, 1])
        else:
            axs[row, 0].scatter(xs, ys)
            axs[row, 1].scatter(xs, ys)

        axs[row, 0].set_axis_off()
        axs[row, 1].set_axis_off()

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)
    
    return f

def train(this_directory,
          model,
          train_stimuli, train_fixations, train_baseline,
          val_stimuli, val_fixations, val_baseline,
          TEM_train_stimuli, TEM_train_fixations, TEM_train_baseline,
          TEM_val_stimuli, TEM_val_fixations, TEM_val_baseline,
          optimizer_config, lr_scheduler_config, minimum_learning_rate,
          #initial_learning_rate, learning_rate_scheduler, learning_rate_decay, learning_rate_decay_epochs, learning_rate_backlook, learning_rate_reset_strategy, minimum_learning_rate,
          batch_size=2,
          ratio_used=0.25,
          validation_metric='IG',
          validation_metrics=['IG', 'LL', 'AUC', 'NSS'],
          iteration_element='image',
          averaging_element='image',
          startwith=None):
    mkdir_p(this_directory)
    
    print("TRAINING DATASET", len(train_fixations.x))
    print("VALIDATION DATASET", len(val_fixations.x))
    
    if os.path.isfile(os.path.join(this_directory, 'final--300-TEM_stepyy_nn.pth')):
        print("Training Already finished")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Using device", device)
    
    model.to(device)
    
    print("optimizer", optimizer_config)
    print("lr_scheduler", lr_scheduler_config)
    
    optimizer_class = import_class(optimizer_config['type'])
    optimizer = optimizer_class(model.parameters(), **optimizer_config['params'])
    
    scheduler_class = import_class(lr_scheduler_config['type'])
    scheduler = scheduler_class(
        optimizer,
        **lr_scheduler_config['params']
    )
    
    if iteration_element == 'image':
        dataset_class = ImageDataset_TEM
    elif iteration_element == 'fixation':
        dataset_class = lambda *args, **kwargs: FixationDataset(*args, **kwargs, included_fixations=model.included_fixations)
    
    print('hey',iteration_element,train_stimuli,train_fixations,'hey')
    train_dataset = dataset_class(train_stimuli,TEM_train_stimuli, train_fixations, train_baseline, TEM_train_baseline, transform=FixationMaskTransform(), average=averaging_element)
    val_dataset = dataset_class(val_stimuli,TEM_val_stimuli ,val_fixations, val_baseline, TEM_val_baseline, transform=FixationMaskTransform(), average=averaging_element)

    #print('wii',train_dataset)    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=ImageDatasetSampler(train_dataset, batch_size=batch_size, ratio_used=ratio_used),
        pin_memory=False,
        num_workers=0,  # doesn't work for sparse tensors yet. Might work soon.
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_sampler=ImageDatasetSampler(val_dataset, batch_size=batch_size),
        pin_memory=False,
        num_workers=0,
    )
    
    if iteration_element == 'image':
        vis_stimuli, vis_fixations, vis_TEM = create_subset_TEM(val_stimuli, TEM_train_stimuli, val_fixations, list(range(batch_size)))
    if iteration_element == 'fixation':
        vis_stimuli, vis_fixations = create_subset(val_stimuli, val_fixations, [0])
        vis_fixations = vis_fixations[:batch_size]
    vis_dataset = dataset_class(vis_stimuli, vis_TEM, vis_fixations, val_baseline, TEM_val_baseline, transform=FixationMaskTransform(), average=averaging_element)
    vis_data_loader = torch.utils.data.DataLoader(
        vis_dataset,
        batch_sampler=ImageDatasetSampler(vis_dataset, batch_size=batch_size, shuffle=False),
        pin_memory=False,
    )
    
    val_metrics = defaultdict(lambda: [])
    
    if startwith is not None:
        restore_from_checkpoint(model, optimizer, scheduler, startwith)
    
    #writer = SummaryWriter(os.path.join(this_directory, 'log'), flush_secs=30)
    
    columns = ['epoch', 'timestamp', 'learning_rate', 'loss']
    for metric in validation_metrics:
        columns.append(f'validation_{metric}')
    
    progress = pd.DataFrame(columns=columns)
    
    step = 0
    last_loss = np.nan
    
    def save_step():

        save_training_state(
            model, optimizer, scheduler, step, last_loss,
            '{}/stepyy-300-plus_qqnTEM_nn_{:03d}.pth'.format(this_directory, step),
        )
        
        #f = visualize(model, vis_data_loader)
        #if is_notebook:
        #    display(f)
        
        with open(os.path.join(this_directory, 'log_opt-TEM_train')+'.csv','a') as fd:
              fd.write('prediction:'+str(step)+',training/loss:'+str(last_loss)+',training/learning_rate:'+str(optimizer.state_dict()['param_groups'][0]['lr'])+',parameters/sigma:'+str(model.finalizer.gauss.sigma.detach().cpu().numpy())+',parameters/center_bias_weight:'+str(model.finalizer.center_bias_weight.detach().cpu().numpy()[0]))
        
        _val_metrics = eval_epoch(model, val_loader, device, val_baseline, TEM_val_baseline, metrics=validation_metrics, averaging_element=averaging_element)
        for key, value in _val_metrics.items():
            val_metrics[key].append(value)

        with open(os.path.join(this_directory, 'log_opt-plus_qqntest_uu_')+'.csv','a') as fd:       
        	for key, value in _val_metrics.items():
            		fd.write('validation/'+str(key)+':'+','+str(value)+','+str(step))
        
        new_row = {
            'epoch': step,
            'timestamp': datetime.utcnow(),
            'learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
            'loss': last_loss,
            #'validation_ig': val_igs[-1]
        }
        for key, value in _val_metrics.items():
            new_row[f'validation/{key}'] = value
        
        progress.loc[step] = new_row

        print(progress.tail(n=2))
        print(progress[['validation_{}'.format(key) for key in val_metrics]].idxmax(axis=0))

        progress.to_csv('{}/log_opt-TEM.csv'.format(this_directory))
        
        for old_step in range(1, step):
            # only check if we are computing validation metrics...
            if val_metrics[validation_metric] and old_step == np.argmax(val_metrics[validation_metric]):
                continue
            for filename in glob.glob('{}/step-plus_TEM_{:03d}.pth'.format(this_directory, old_step)):
                print("removing", filename)
                os.remove(filename)


    old_checkpoints = sorted(glob.glob(os.path.join(this_directory, 'step-plus_TEM_*.pth')))
    if old_checkpoints:
        last_checkpoint = old_checkpoints[-1]
        print("Found old checkpoint", last_checkpoint)
        step, last_loss = restore_from_checkpoint(model, optimizer, scheduler, last_checkpoint)
        print("Setting step to", step)

    if step == 0:
        print("Beginning training")
        save_step()

    else:
        print("Continuing from step", step)
        progress = pd.read_csv(os.path.join(this_directory, 'log_opt-TEM.csv'), index_col=0)
        val_metrics = {}
        for column_name in progress.columns:
            if column_name.startswith('validation_'):
                val_metrics[column_name.split('validation_', 1)[1]] = list(progress[column_name])

        if step not in progress.epoch.values:
            print("Epoch not yet evaluated, evaluating...")
            save_step()

        print(progress)

    while optimizer.state_dict()['param_groups'][0]['lr'] >= minimum_learning_rate:
        step += 1
        last_loss = train_epoch(model, train_loader, optimizer, device)
        #gpu_profile(frame=sys._getframe(), event='line', arg=None)
        save_step()
        scheduler.step()
    
    torch.save(model.state_dict(), '{}/final--TEM.pth'.format(this_directory))
    
    for filename in glob.glob(os.path.join(this_directory, 'step-plus_TEM_*')):
        print("removing", filename)
        os.remove(filename)


def _get_from_config(key, *configs, **kwargs):
    """get config keys with fallbacks"""
    for config in configs:
        try:
            print(glom(config,key,**kwargs))
            return glom(config, key, **kwargs)
        except KeyError:
            pass
    raise KeyError(key, configs)

assert _get_from_config('a.b', {'a': {'c': 1}}, {'a': {'b': 2}}) == 2


def _get_stimulus_filename(stimuli_stimulus):
    stimuli = stimuli_stimulus.stimuli
    index = stimuli_stimulus.index
    if isinstance(stimuli, pysaliency.FileStimuli):
        return stimuli.filenames[index]
    elif isinstance(stimuli, pysaliency.datasets.ObjectStimuli):
        return _get_stimulus_filename(stimuli.stimulus_objects[index])
    else:
        raise TypeError("Stimuli of type {} don't have filenames!".format(type(stimuli)))

def get_filenames_for_stimuli(stimuli):
    if isinstance(stimuli, pysaliency.datasets.FileStimuli):
        return list(stimuli.filenames)
    if isinstance(stimuli, pysaliency.datasets.ObjectStimuli):
        return [_get_stimulus_filename(s) for s in stimuli.stimulus_objects]

def make_file_stimuli(stimuli):
    return pysaliency.datasets.FileStimuli(get_filenames_for_stimuli(stimuli))

def _get_dataset(dataset_config, training_config=None, string_indicator=None):
    """return stimuli, fixations, centerbias"""
    centerbias = None
    #print(isinstance(dataset_config, str),'wii')
    #print(dataset_config)
    if isinstance(dataset_config, str):
        print(string_indicator)
        dataset_config = { 'name':  dataset_config, 'stimuli':training_config['train_dataset'],'stimuli_TEM_val':training_config['TEM_dataset_val'],'stimuli_TEM':training_config['TEM_dataset'], 'train_dataset': training_config['train_dataset'], 'fixations_val':training_config['val_fixations'] , 'fixations': training_config['fixations'],
                          'filters': []}
    #print(dataset_config['stimuli']) 
    if string_indicator == 'train':
        stimuli = pysaliency.external_datasets.read_hdf5(dataset_config['stimuli'])
        stimuli_TEM=pysaliency.external_datasets.read_hdf5(dataset_config['stimuli_TEM'])
        fixations = pysaliency.external_datasets.read_hdf5(dataset_config['fixations'])
        fixations_TEM = pysaliency.external_datasets.read_hdf5(dataset_config['fixations'])
    
    if string_indicator == 'val':
        dataset_config['stimuli']=dataset_config['name']
        dataset_config['fixations']=root_directory+'/fixations_val.hdf5'
        
        stimuli = pysaliency.external_datasets.read_hdf5(dataset_config['stimuli'])
        fixations = pysaliency.external_datasets.read_hdf5(dataset_config['fixations'])
        stimuli_TEM=pysaliency.external_datasets.read_hdf5(dataset_config['stimuli_TEM_val'])
        fixations_TEM = pysaliency.external_datasets.read_hdf5(dataset_config['fixations'])
    
    if string_indicator == 'test':
        dataset_config['stimuli']=dataset_config['name']
        dataset_config['fixations']=root_directory+'/fixations_train.hdf5'
        
        stimuli = pysaliency.external_datasets.read_hdf5(dataset_config['stimuli'])
        fixations = pysaliency.external_datasets.read_hdf5(dataset_config['fixations'])
        stimuli_TEM=pysaliency.external_datasets.read_hdf5(dataset_config['stimuli_TEM_val'])
        fixations_TEM = pysaliency.external_datasets.read_hdf5(dataset_config['fixations'])    
    

    if string_indicator == 'train':
    	pysaliency_config = dict(dataset_config)
    	centerbias_file = pysaliency_config.pop('centerbias', None)
    	stimuli, fixations = load_dataset_from_config(pysaliency_config)
     
    ## define the centerbias and stimuli files depending the information you want to use for training, calculate them offline

    if string_indicator == 'train':
        centerbias_path = root_directory+'/centerbias_train.hdf5'
        centerbias_TEM_path = root_directory+'/centerbias_train_TEM_dist_n.hdf5'
        #_get_from_config('centerbias', dataset_config, training_config)
    if string_indicator == 'val':
        #centerbias_path = '/gpfs/scratch/jmayortorres/deepgaze_pytorch-master/centerbias_optimized_cross.hdf5'
        centerbias_path = root_directory+'/centerbias_val.hdf5'
        centerbias_TEM_path = root_directory+'/centerbias_val_TEM_dist_n.hdf5'
    if string_indicator == 'test':
        centerbias_path = root_directory+'/centerbias_val.hdf5'     
        centerbias_TEM_path = root_directory+'/centerbias_val_TEM_dist_n.hdf5'

    centerbias = HDF5Model(stimuli, centerbias_path)
    centerbias_TEM = HDF5Model(stimuli_TEM, centerbias_TEM_path)

    return stimuli, fixations, centerbias, stimuli_TEM, fixations_TEM, centerbias_TEM

def iterate_crossvalidation_config(stimuli, fixations, crossval_config):
    for fold_no, (train_stimuli, train_fixations, val_stimuli, val_fixations, test_stimuli, test_fixations)             in enumerate(iterate_crossvalidation(
                stimuli, fixations,
                crossval_folds=crossval_config['folds'],
                val_folds=crossval_config['val_folds'],
                test_folds=crossval_config['test_folds']
            )):
        
        yield crossval_config["folds"], fold_no, train_stimuli, train_fixations, val_stimuli, val_fixations, test_stimuli, test_fixations


def run_training_part(training_config, full_config, final_cleanup=False):
    print("Running training part", training_config['name'])
    print("Configuration of this training part:")
    print(yaml.safe_dump(training_config))
    
    if 'val_dataset' in training_config and 'crossvalidation' in training_config:
        raise ValueError("Cannot specify both validation dataset and crossvalidation")
    
    directory = os.path.join(root_directory, training_config['name'])

    training_config['train_dataset']=root_directory+'/stimuli_train.hdf5'
    training_config['TEM_dataset']=root_directory+'/stimuli_train_TEM_dist_n.hdf5'
    training_config['TEM_dataset_val']=root_directory+'/stimuli_val_TEM_dist_n.hdf5'
    training_config['fixations']=root_directory+'/fixations_train.hdf5'
    training_config['val_dataset']=root_directory+'/stimuli_val.hdf5'
    training_config['test_dataset']=root_directory+'/stimuli_val.hdf5'
    training_config['val_fixations']=root_directory+'/fixations_val.hdf5'
    training_config['test_fixations']=root_directory+'/fixations_val.hdf5'
    train_stimuli, train_fixations, train_centerbias, TEM_train_stimuli, TEM_train_fixations, TEM_train_centerbias = _get_dataset(training_config['train_dataset'],training_config,'train')

    if 'val_dataset' in training_config:
        val_stimuli, val_fixations, val_centerbias, TEM_val_stimuli, TEM_val_fixations, TEM_val_centerbias = _get_dataset(training_config['val_dataset'], training_config,'val')
        if 'test_dataset' in training_config:
            test_stimuli, test_fixations, test_centerbias, TEM_test_stimuli, TEM_test_fixations, TEM_test_centerbias = _get_dataset(training_config['test_dataset'], training_config,'test')
        else:
            test_stimuli = test_fixations = test_centerbias = TEM_test_stimuli = TEM_test_fixations = TEM_test_centerbias = None
        
        def iter_fn():
            return [{
                'config': training_config,
                'directory': directory,
                'fold_no': None,
                'crossval_folds': None,
                'train_stimuli': train_stimuli,
                'train_fixations': train_fixations,
                'train_centerbias': train_centerbias,
                'val_stimuli': val_stimuli,
                'val_fixations': val_fixations,
                'val_centerbias': val_centerbias,
                'train_stimuli_TEM': TEM_train_stimuli,
                'train_fixations_TEM': TEM_train_fixations,
                'train_centerbias_TEM': TEM_train_centerbias,
                'val_stimuli_TEM': TEM_val_stimuli,
                'val_fixations_TEM': TEM_val_fixations,
                'val_centerbias_TEM': TEM_val_centerbias,
                'test_stimuli_TEM': test_stimuli,
                'test_fixations': test_fixations,
                'test_centerbias': test_centerbias,
                'test_stimuli_TEM': TEM_test_stimuli,
                'test_fixations_TEM': TEM_test_fixations,
                'test_centerbias_TEM': TEM_test_centerbias,
            }]
        
    else:
        assert 'crossvalidation' in training_config
        def iter_fn():
            for crossval_folds, fold_no, _train_stimuli, _train_fixations, _val_stimuli, _val_fixations, _test_stimuli, _test_fixations in iterate_crossvalidation_config(train_stimuli, train_fixations, training_config['crossvalidation']):
                yield {
                    'config': training_config,
                    'directory': os.path.join(directory, f'crossval-{crossval_folds}-{fold_no}'),
                    'fold_no': fold_no,
                    'crossval_folds': crossval_folds,
                    'train_stimuli': _train_stimuli,
                    'train_fixations': _train_fixations,
                    'train_centerbias': train_centerbias,
                    'val_stimuli': _val_stimuli,
                    'val_fixations': _val_fixations,
                    'val_centerbias': train_centerbias,
                    'test_stimuli': _test_stimuli,
                    'test_fixations': _test_fixations,
                    'test_centerbias': train_centerbias,
                }
    
    for part in iter_fn():
        
        if args.crossval_fold_number is not None and part['fold_no'] != args.crossval_fold_number:
            print("Skipping crossval fold number", part['fold_no'])
            continue
        
        model = build_model(training_config['model'])
        
        startwith = part['config']['startwith']
        if startwith is not None:
            startwith = startwith.format(
                root_directory=root_directory,
                crossval_folds=part['crossval_folds'],
                fold_no=part['fold_no']
            )
                
        train(
            this_directory=os.getcwd(),
            model=model,
            train_stimuli=part['train_stimuli'],
            train_fixations=part['train_fixations'],
            train_baseline=part['train_centerbias'],
            val_stimuli=part['val_stimuli'],
            val_fixations=part['val_fixations'],
            val_baseline=part['val_centerbias'],
            TEM_train_stimuli=part['train_stimuli_TEM'], 
            TEM_train_fixations=part['train_fixations_TEM'], 
            TEM_train_baseline=part['train_centerbias_TEM'],
            TEM_val_stimuli=part['val_stimuli_TEM'], 
            TEM_val_fixations=part['val_fixations_TEM'], 
            TEM_val_baseline=part['val_centerbias_TEM'],
            optimizer_config=part['config']['optimizer'],
            lr_scheduler_config=part['config']['lr_scheduler'],
            minimum_learning_rate=part['config']['minimal_learning_rate'],
            startwith=startwith,
            iteration_element=part['config']['iteration_element'],
            averaging_element=part['config']['averaging_element'],
            batch_size=part['config']['batch_size'],
            ratio_used=part['config']['training_dataset_ratio_per_epoch'],
            validation_metrics=part['config']['validation_metrics'],
            validation_metric=part['config']['validation_metric'],
        )
    
    if final_cleanup:
        run_cleanup(iter_fn, directory, training_config['final_cleanup'])
        return
    
    if training_config['evaluation']:
        run_evaluation(iter_fn, directory, training_config)
    
    if training_config['cleanup']:
        run_cleanup(iter_fn, directory, training_config['cleanup'])


def run_evaluation(iter_fn, directory, training_config):
    evaluation_config = training_config['evaluation']
    if evaluation_config['compute_metrics']:
        compute_metrics(iter_fn, directory, evaluation_config['compute_metrics'], training_config)
    if evaluation_config['compute_predictions']:
        compute_predictions(iter_fn, directory, evaluation_config['compute_predictions'], training_config)

def _get_dataset_for_part(part, dataset):
    if dataset == 'training':
        return part['train_stimuli'], part['train_fixations'], part['train_centerbias']
    if dataset == 'validation':
        return part['val_stimuli'], part['val_fixations'], part['val_centerbias']
    if dataset == 'test':
        return part['test_stimuli'], part['test_fixations'], part['test_centerbias']
    raise ValueError(dataset)
        
def compute_metrics(iter_fn, directory, evaluation_config, training_config):
    metrics = []
    weights = {dataset: [] for dataset in evaluation_config['datasets']}
    
    results_file = os.path.join('/scratch/c.sapjm10/deepgaze_master_Evaluation/results_TEM.csv')
    if not os.path.isfile(results_file):
        for part in iter_fn():
            this_directory = os.getcwd()

            if os.path.isfile(os.path.join('/scratch/c.sapjm10/deepgaze_master_Evaluation/results_TEM.csv')):
                results = pd.read_csv(os.path.join(this_directory, 'results_TEM.csv'), index_col=0)
                
                metrics.append(results)
                
                for dataset in evaluation_config['datasets']:
                    if dataset == 'validation':
                    	_stimuli, _fixations, _ = _get_dataset(training_config['val_dataset'], training_config,'train') #_get_dataset_for_part(part, dataset)
                    if dataset == 'test':
                        _stimuli, _fixations, _ = _get_dataset(training_config['test_dataset'], training_config,'val') #_get_dataset_for_part(part, dataset)
                    weights[dataset].append(len(_fixations.x))

                continue

            
            model = build_model(training_config['model'])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Using device", device)
            model.to(device)
            
            restore_from_checkpoint(model, None, None, os.path.join(os.getcwd(),'final--300-TEM.pth'))
    
        
            this_results = {metric: {} for metric in evaluation_config['metrics']}
            
            for dataset in evaluation_config['datasets']:

                if dataset == 'training':
                        _stimuli, _fixations, _centerbias, _stim_TEM, _stim_TEM_fixations, _stim_TEM_centerbias = _get_dataset(training_config['train_dataset'], training_config,'train') #_get_dataset_for_part(part, dataset)
                if dataset == 'validation':
                        _stimuli, _fixations, _centerbias, _stim_TEM, _stim_TEM_fixations, _stim_TEM_centerbias = _get_dataset(training_config['val_dataset'], training_config,'val') #_get_dataset_for_part(part, dataset)
                if dataset == 'test':
                        _stimuli, _fixations, _centerbias, _stim_TEM, _stim_TEM_fixations, _stim_TEM_centerbias = _get_dataset(training_config['test_dataset'], training_config,'test') #_get_dataset_for_part(part, dataset)
                #_stimuli, _fixations, _centerbias = #_get_dataset_for_part(part, dataset)
                
                
                if part['config']['iteration_element'] == 'image':
                    dataset_class = ImageDataset_TEM
                elif part['config']['iteration_element'] == 'fixation':
                    dataset_class = lambda *args, **kwargs: FixationDataset(*args, **kwargs, included_fixations=model.included_fixations)
                else:
                    raise ValueError(part['config']['iteration_element'])

                _dataset = dataset_class(_stimuli, _stim_TEM, _fixations, _centerbias, _stim_TEM_centerbias, transform=FixationMaskTransform(), average=part['config']['averaging_element'])
                loader = torch.utils.data.DataLoader(
                    _dataset,
                    batch_sampler=ImageDatasetSampler(_dataset, batch_size=part['config']['batch_size'], shuffle=False),
                    pin_memory=False,
                    num_workers=0,  # doesn't work for sparse tensors yet. Might work soon.
                )
                
                _results = eval_epoch(model, loader, device, _centerbias, _stim_TEM_centerbias, metrics=evaluation_config['metrics'], averaging_element=part['config']['averaging_element'],name=dataset)
                
                for metric in evaluation_config['metrics']:
                    this_results[metric][dataset] = _results[metric]
                
                if part['config']['averaging_element'] == 'fixation':
                    weights[dataset].append(len(_fixations.x))
                elif part['config']['averaging_element'] == 'image':
                    weights[dataset].append(len(_stimuli))
            
            result_df = pd.DataFrame(this_results, columns=evaluation_config['metrics']).loc[evaluation_config['datasets']]
            result_df.to_csv(os.path.join(this_directory, 'results_TEM.csv'))
            
            metrics.append(result_df)

        rows = []
        for dataset in evaluation_config['datasets']:
            _weights = weights[dataset]
            relative_weights = np.array(_weights) / np.array(_weights).sum()
            _results = [df.loc[dataset] for df in metrics]
            _result = sum(weight*df for weight, df in zip(relative_weights, _results))
            rows.append(_result)
        
        result_df = pd.DataFrame(rows)

        result_df.to_csv(results_file)

        print(result_df)
    else:
        print(pd.read_csv(results_file, index_col=0))

        
class SharedPyTorchModel(object):
    def __init__(self, model):
        self.model = model
        self.active_checkpoint = None

    def load_checkpoint(self, checkpoint_path):
        if self.active_checkpoint != checkpoint_path:
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.active_checkpoint = checkpoint_path


class DeepGazeCheckpointModel(models.Model):
    def __init__(self, shared_model, checkpoint, centerbias_model, TEM_model, centerbias_TEM):
        super().__init__(caching=False)

        self.checkpoint = checkpoint
        self.centerbias_model = centerbias_model
        self.centerbias_model_TEM = TEM_model
        self.shared_model = shared_model
        self.stim_TEM = TEM_model
        self.centerbias_TEM = centerbias_TEM

    def _log_density(self, stimulus,stim_TEM):
        self.shared_model.load_checkpoint(self.checkpoint)
        images = torch.tensor([stimulus.transpose(2, 0, 1)], dtype=torch.float64)
        stim_TEM = torch.tensor([self.stim_TEM.transpose(2, 0, 1)], dtype=torch.float64)
        centerbiases = torch.tensor([self.centerbias_model.log_density(stimulus)], dtype=torch.float64)
        centerbiases_TEM = torch.tensor([self.centerbias_TEM.log_density(stimulus)], dtype=torch.float64)
        log_density = self.shared_model.model.forward(images.type('torch.FloatTensor'),stim_TEM.type('torch.FloatTensor'), centerbiases.type('torch.FloatTensor'),centerbiases_TEM.type('torch.FloatTensor'))[0, :, :].detach().cpu().numpy()
        
        return log_density

    def _log_density_n(self, stimulus,stim_TEM):
        self.shared_model.load_checkpoint(self.checkpoint)
        images = torch.tensor([stimulus.transpose(2, 0, 1)], dtype=torch.float64)
        stim_TEM_n = torch.tensor([stim_TEM.transpose(2, 0, 1)], dtype=torch.float64)
        centerbiases = torch.tensor([self.centerbias_model.log_density(stimulus)], dtype=torch.float64)
        centerbiases_TEM = torch.tensor([self.centerbias_TEM.log_density(stim_TEM)], dtype=torch.float64)
        log_density = self.shared_model.model.forward(images.type('torch.FloatTensor'),stim_TEM_n.type('torch.FloatTensor'), centerbiases.type('torch.FloatTensor'),centerbiases_TEM.type('torch.FloatTensor'))[0, :, :].detach().cpu().numpy()

        return log_density

        
def compute_predictions(iter_fn, directory, prediction_config, training_config):
    
    model = build_model(training_config['model'])
    shared_model = SharedPyTorchModel(model)
    bashCommand1 = "cp vgg.py vgg_temp.py"
    bashCommand2 = "cp vgg_pred.py vgg.py"
    process1 = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE)
    output1, error1 = process1.communicate()
    process2 = subprocess.Popen(bashCommand2.split(), stdout=subprocess.PIPE)
    output2, error2 = process2.communicate()
    
    for dataset in prediction_config['datasets']:
        print(f"Computing predictions for dataset {dataset}")
        models = {}
        dataset_stimuli = []
        dataset_fixations = []
        data_TEM = []
        for part in iter_fn():
            this_directory = part['directory']
            checkpoint = os.path.join(this_directory, 'final--300-TEM.py')
            if dataset == 'training':
                _stimuli, _fixations, _centerbias, _stimuli_TEM, _fixations_TEM, _centerbias_TEM = _get_dataset(training_config['train_dataset'], training_config,'train')
            if dataset == 'validation':
            	_stimuli, _fixations, _centerbias, _stimuli_TEM, _fixations_TEM, _centerbias_TEM = _get_dataset(training_config['val_dataset'], training_config,'train')
            if dataset == 'test':
                _stimuli, _fixations, _centerbias, _stimuli_TEM, _fixations_TEM, _centerbias_TEM = _get_dataset(training_config['val_dataset'], training_config,'val')
            models[_stimuli] = DeepGazeCheckpointModel(shared_model, checkpoint, _centerbias, _stimuli_TEM,_centerbias_TEM)
            dataset_stimuli.append(_stimuli)
            dataset_fixations.append(_fixations)
            data_TEM.append(_stimuli_TEM)

        model = pysaliency.models.StimulusDependentModel(models)
        stimuli, fixations = pysaliency.datasets.concatenate_datasets(dataset_stimuli, dataset_fixations)
        TEM, fixations = pysaliency.datasets.concatenate_datasets(data_TEM, dataset_fixations)
        file_stimuli = make_file_stimuli(stimuli)
        file_TEM = make_file_stimuli(TEM)
        pysaliency.precomputed_models.export_model_to_hdf5_n(model,file_stimuli,file_TEM,os.path.join(os.getcwd(),f'{dataset}_predictions_SALICON_baseline.hdf5'))

        
def run_cleanup(iter_fn, directory, cleanup_config):
    if cleanup_config['cleanup_checkpoints']:
        for part in iter_fn():
            this_directory = part['directory']
            for filename in glob.glob(os.path.join(this_directory, '*.ckpt.*')):
                print("removing", filename)
                os.remove(filename)


def test_crossval_split():
    print(get_crossval_folds(
            crossval_folds=10,
            crossval_no=1,
            val_folds=1,
            test_folds=1,
    ))



'''
import linecache

os.environ['CUDA_LAUNCH_BLOCKING']='1'

import pynvml3
from py3nvml import py3nvml
import torch
import socket


if 'GPU_DEBUG' in os.environ:
    gpu_profile_fn = f"Host_{socket.gethostname()}_gpu{os.environ['GPU_DEBUG']}_mem_prof-{datetime.datetime.now():%d-%b-%y-%H-%M-%S}.prof.txt"
    print('profiling gpu usage to ', gpu_profile_fn)


## Global variables
last_tensor_sizes = set()
last_meminfo_used = 0
lineno = None
func_name = None
filename = None
module_name = None


def gpu_profile(frame, event, arg):
    # it is _about to_ execute (!)
    global last_tensor_sizes
    global last_meminfo_used
    global lineno, func_name, filename, module_name

    if event == 'line':
        try:
            # about _previous_ line (!)
            if lineno is not None:
                py3nvml.nvmlInit()
                handle = py3nvml.nvmlDeviceGetHandleByIndex(int(os.environ['GPU_DEBUG']))
                meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
                line = linecache.getline(filename, lineno)
                where_str = module_name+' '+func_name+':'+str(lineno)

                new_meminfo_used = meminfo.used
                mem_display = new_meminfo_used-last_meminfo_used if use_incremental else new_meminfo_used
                with open(gpu_profile_fn, 'a+') as f:
                    f.write(f"{where_str:<50}"
                            f":{(mem_display)/1024**2:<7.1f}Mb "
                            f"{line.rstrip()}\n")

                    last_meminfo_used = new_meminfo_used
                    if print_tensor_sizes is True:
                        for tensor in get_tensors():
                            if not hasattr(tensor, 'dbg_alloc_where'):
                                tensor.dbg_alloc_where = where_str
                        new_tensor_sizes = {(type(x), tuple(x.size()), x.dbg_alloc_where)
                                            for x in get_tensors()}
                        for t, s, loc in new_tensor_sizes - last_tensor_sizes:
                            f.write(f'+ {loc:<50} {str(s):<20} {str(t):<10}\n')
                        for t, s, loc in last_tensor_sizes - new_tensor_sizes:
                            f.write(f'- {loc:<50} {str(s):<20} {str(t):<10}\n')
                        last_tensor_sizes = new_tensor_sizes
                py3nvml.nvmlShutdown()

            # save details about line _to be_ executed
            lineno = None

            func_name = frame.f_code.co_name
            filename = frame.f_globals["__file__"]
            if (filename.endswith(".pyc") or
                    filename.endswith(".pyo")):
                filename = filename[:-1]
            module_name = frame.f_globals["__name__"]
            lineno = frame.f_lineno
            
            #only profile codes within the parenet folder, otherwise there are too many function calls into other pytorch scripts
            #need to modify the key words below to suit your case.
            if 'gpu_memory_profiling' not in os.path.dirname(os.path.abspath(filename)):   
                lineno = None  # skip current line evaluation

            if ('car_datasets' in filename
                    or '_exec_config' in func_name
                    or 'gpu_profile' in module_name
                    or 'tee_stdout' in module_name):
                lineno = None  # skip othe unnecessary lines
            
            return gpu_profile

        except (KeyError, AttributeError):
            pass

    return gpu_profile


def get_tensors(gpu_only=True):
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda:
                yield tensor
        except Exception as e:
            pass
'''

for training_part in config['training']['parts']:
    if args.training_part is not None and training_part['name'] != args.training_part:
        print("Skipping part", args.training_part)
        continue
    run_training_part(training_part, config)

