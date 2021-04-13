from tqdm import tqdm
import torch
import os
import glob
import numpy as np
import pysaliency.datasets_old_old as datasets
from scipy.io import loadmat
temp_dir = 'COCO_subfolder_output'
location = ''  # define here the location for your output .hdf5 files
fixations = []
fixation_type = 'fixations'
if fixation_type == 'mouse':
    fixation_attr = 'location'
elif fixation_type == 'fixations':
    fixation_attr = 'fixations'

for dataset in ['train', 'val']:
    ns = []
    train_xs = []
    train_ys = []
    train_ts = []
    train_subjects = []

    subject_id = 0
    print(os.path.join(temp_dir, 'fixations', dataset, '*.mat'))
    data_files = list(
        sorted(glob.glob(os.path.join(temp_dir, 'fixations', dataset, '*.mat'))))
    for n, filename in enumerate(tqdm(data_files)):
        fixation_data = loadmat(filename)
        for subject_data in fixation_data['gaze'].flatten():
            # matlab: one-based indexing
            train_xs.append(subject_data[fixation_attr][:, 0] - 1)
            train_ys.append(subject_data[fixation_attr][:, 1] - 1)
            if fixation_type == 'mouse':
                train_ts.append(subject_data['timestamp'].flatten())
            elif fixation_type == 'fixations':
                train_ts.append(range(len(train_xs[-1])))
            train_subjects.append(
                np.ones(len(train_xs[-1]), dtype=int)*subject_id)
            ns.append(np.ones(len(train_xs[-1]), dtype=int)*n)
            subject_id += 1

    train_xs = np.hstack(train_xs)
    train_ys = np.hstack(train_ys)
    train_ts = np.hstack(train_ts)
    train_subjects = np.hstack(train_subjects)
    ns = np.hstack(ns)
    # use this declaration if you are using FixationTrains object instead of Fixation, make sure the pysaliency package is updated
    fixations.append(datasets.FixationTrains.from_fixation_trains(torch.Tensor(train_xs.astype(dtype='float32')), torch.Tensor(
        train_ys.astype(dtype='float32')), torch.Tensor(train_ts.astype(dtype='float32')), torch.Tensor(ns), train_subjects))
    del train_xs, train_ys, train_ts, train_subjects, ns

fixations_train, fixations_val = fixations

fixations_train.to_hdf5(os.path.join(location, 'fixations_train_train.hdf5'))
fixations_val.to_hdf5(os.path.join(location, 'fixations_val_train.hdf5'))
