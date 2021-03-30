## DeepGaze adding Text-Embedding-Map features
This repository includes the implementation of **DeepGaze** adding the **Text-Embedding-Maps** (TEM) [Barman et., al 2020](https://arxiv.org/abs/2002.06144) - [Yang et., al 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Yang_Learning_to_Extract_CVPR_2017_paper.html) for predicting robustly human fixation/gaze.

We used a data intersection between [COCO](https://cocodataset.org/#home) and [SALICON](http://salicon.net/) to perform our evaluations using the fixations from SALICON and the panoptic annotations/segmentations from COCO.

First download the image data from the following links:

[COCO images](https://drive.google.com/uc?export=download&id=1RM4gXlSIic22HvYHaS5XOGmjcLDSUiUv)

For the fixation data use the hdf5 files included in the **experiments_root/** folder

and the COCO panoptic segmentations from here and unzip them **use the train and val panoptic sets released on 2017**:

[COCO download**](https://cocodataset.org/#download)


unzip the file and allocate the folders in the right places you will use it for run the code and modifify from the code if youn need it
```bash
unzip COCO_subfolder_output.zip
```
First generate the TEM, take into account that the file **file_annotations/sal_ground_truth_emb_SALICON_TEM_w.txt** if you want to generate the embedding from scratch first define your new training folder and run:
```python
python generate_TEM/generate_objects_co_occur.py
```
This will generate a new co-occurrences matrix in a file called **sal_cooccur_mat_new.txt**. To obtain the new embeddings use the instructions on the package **[Mittens](https://github.com/roamanalytics/mittens)** and upload the file **sal_cooccur_mat_new.txt** as csv.

Having your embedding file you can create your TEM images assigning an output folder for them and running

```python
python generate_TEM/generate_TEM_train.py
python generate_TEM/generate_TEM_val.py
```
The previous calls will generate the TEM images on .tiff format including the 300 dimension of the semantic spaces obtained by Mittens substracting the semantic space of the annotated scene and the objects in the scene. 

If you want to do the same but creating some TEM images with the Cosine, Euclidean, and Chebyshev distances between the annotated scene and the objectes run:
```python
python generate_TEM/generate_TEM_train_dist.py
python generate_TEM/generate_TEM_val_dist.py
```

Now you must create the **centerbias** files for the stimuli and the TEM images. For doing that be sure the stimuli_train.hdf5, fixations_train.hdf5, stimuli_val.hdf5, fixations_val.hdf5, stimuli_TEM_train.hdf5, and stimuli_TEM_val.hdf5 files are located in the **experiment_root/** folder. Then, you can run: 
```python
python create_centerbias.py
python create_centerbias_TEM.py
```
Now, you are ready to the training! and you can run this:
```python
python run_dgII_evaluation_plus_TEM.py
```
log.txt files will present you the current status of the learning, or if you want to wait until the end of the training  the file **results_TEM.csv** will show you the final Log-likehood (LL), Information Gain (IG), Area Under Curve (AUC), and Normalized Scanpath Saliency (NSS).

A LL evolution through the training epochs could be observed in the following Figure including TEM features. An [AdaBound](https://github.com/Luolc/AdaBound) optimizer and a final Drop-out layer (before the Finalizer) must be added to the network for avoiding overfitting. The full pipeline of out semantic-based gaze prediction is shown in the following Figure:

<img src="https://github.com/meiyor/DeepGaze-Text-Embedding-Map/blob/main/pipeline_def_new_no_scan.jpg" width="1100" height="300">

The performance comparison between the [DeepGaze](https://github.com/matthias-k/deepgaze_pytorch) baseline and the DeepGaze+TEM is the following:

|   | **IG** | **LL** | **AUC** | **NSS** | 
| ------------- | ------------- |  ------------- | ------------- |  ------------- |
| **DeepGaze II**  | 0.18521176210720047 | 0.6865235706511486 | 0.7519319139677038 | 1.043845942603672 |
| **DeepGaze + TEM**  | **0.19025571295039756** | **0.6915675214943456** | 0.7522227225707403 | 1.043975352729339 | 
