import glob
from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel
import pysaliency
import pysaliency.external_datasets as external_datasets
import os

with pysaliency.utils.TemporaryDirectory(cleanup=True) as temp_dir:
   temp_dir='TEM_output'
   location='deepgaze_master_Evaluation'
   stimuli_train = external_datasets.create_stimuli(
            stimuli_location = os.path.join('TEM_folder'),
            filenames = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(temp_dir, 'TEM_folder/', 'COCO_train*')))],
            location=os.path.join(location,'datasets/stim_train_TEM') if location else None
        )
   stimuli_val = external_datasets.create_stimuli(
            stimuli_location = os.path.join(temp_dir, 'TEM_val_pca_TEM_100'),
            filenames = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(temp_dir, 'TEM_val_pca_TEM_100/', 'COCO_val*')))],
            location=os.path.join(location, 'datasets/stim_val_TEM0') if location else None
       )
   stimuli_train.to_hdf5(os.path.join(location, 'stimuli_train_TEM_pca_100.hdf5'))
   stimuli_val.to_hdf5(os.path.join(location, 'stimuli_val_TEM_pca_100.hdf5'))
