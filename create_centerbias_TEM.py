from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel
import pysaliency
import pysaliency.external_datasets as external_datasets

stimuli = pysaliency.datasets.read_hdf5('stimuli_train_TEM.hdf5')
fixations = pysaliency.datasets.read_hdf5('fixations_train.hdf5')

center_bias_corrected=BaselineModel(stimuli=stimuli,fixations=fixations,bandwidth=10**-1.6715425643341146,eps=10**-3.619089786523399)
pysaliency.precomputed_models.export_model_to_hdf5(center_bias_corrected,stimuli,filename='centerbias_train_TEM.hdf5')

stimuli = pysaliency.datasets.read_hdf5('/scratch/c.sapjm10/deepgaze_master_Evaluation/stimuli_val_TEM.hdf5')
fixations = pysaliency.datasets.read_hdf5('/scratch/c.sapjm10/deepgaze_master_Evaluation/fixations_val.hdf5')

center_bias_corrected=BaselineModel(stimuli=stimuli,fixations=fixations,bandwidth=10**-1.6715425643341146,eps=10**-3.619089786523399)
pysaliency.precomputed_models.export_model_to_hdf5(center_bias_corrected,stimuli,filename='/scratch/c.sapjm10/deepgaze_master_Evaluation/centerbias_val_TEM.hdf5')
