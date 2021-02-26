## DeepGaze adding Text-Embedding-Map features
This repository includes the implementation of **DeepGaze** adding the **Text-Embedding-Maps** (TEM) [Barman et., al 2020,Yang et., al 2017] for predicting robustly human fixation/gaze.

We used a data intersection between [COCO](https://cocodataset.org/#home) and [SALICON](https://cocodataset.org/#home) to perform our evaluations using the fixations from SALICON and the panoptic annotations/segmentations from COCO.

First download the data from the following link:

[https://drive.google.com/uc?export=download&id=1RM4gXlSIic22HvYHaS5XOGmjcLDSUiUv](https://drive.google.com/uc?export=download&id=1RM4gXlSIic22HvYHaS5XOGmjcLDSUiUv)

and the COCO panoptic segmentations from here and unzip them:

[images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip](images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip)


unzip the file and allocate the folders in the right places you will use it for run the code
```bash
unzip COCO_subfolder_output.zip
```
First generate the TEM, take into account that the file **file_annotations/sal_ground_truth_emb_SALICON_TEM_w.txt** if you want to generate the embedding from scratch first define your new training folder and run:
```python
python generate_TEM/generate_objects_co_occur.py
```
This will generate a new co-occurrences matrix in a file called **sal_cooccur_mat_new.txt**. To obtain the new embeddings use the instructions on the package **[Mittens](https://github.com/roamanalytics/mittens)** and upload the file **sal_cooccur_mat_new.txt** as csv.

Having your embedding calculated you can create your TEM images assigning an output folder for them and running

```python
python generate_TEM/generate_TEM_train_300.py
python generate_TEM/generate_TEM_val_300.py
```
The previous calls will generate the TEM images on .tiff format including the 300 dimension of the semantic spaces obtained by Mittens substracting the semantic space of the annotated scene and the objects in the scene. 

If you want to do the same but creating some TEM images with the Cosine, Euclidean, and Chebyshev distances between the annotated scene and the objectes run:
```python
python generate_TEM/generate_TEM_train_dist.py
python generate_TEM/generate_TEM_val_dist.py
```

Now you must create the **centerbias** files for the stimuli and the TEM images. For doing that be sure the stimuli_train.hdf5, fixations_train.hdf5, stimuli_val.hdf5, fixations_val.hdf5, stimuli_TEM_train.hdf5, and stimuli_TEM_val.hdf5 files are located in the **experiment_root/** folder. Then, you can run: 
