## DeepGaze adding Text-Embedding-Map features
This repository includes the implementation of **DeepGaze** adding the **Text-Embedding-Maps** (TEM) [Barman et., al 2020](https://arxiv.org/abs/2002.06144) - [Yang et., al 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Yang_Learning_to_Extract_CVPR_2017_paper.html) for predicting robustly human fixation/gaze.

We used a data intersection between [COCO](https://cocodataset.org/#home) and [SALICON](http://salicon.net/) to perform our evaluations using the fixations from SALICON and the panoptic annotations/segmentations from COCO.

First install all the dependencies running the following command, please be sure you have pip installed and updated on your bash entry before run this:
```bash
pip install requirements.txt
pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/cocodataset/panopticapi.git
pip install -U mittens
pip install random
```
Subsequently, download the image data from the following links:

[COCO images](https://drive.google.com/u/0/uc?id=1RM4gXlSIic22HvYHaS5XOGmjcLDSUiUv&export=download)

For the fixation data use the hdf5 files included in the **experiments_root/** folder

and the COCO panoptic segmentations from here and unzip them - **use the train and val panoptic sets released on 2017**:

[COCO download**](https://cocodataset.org/#download)


unzip the file and allocate the folders in the right places you will use it for run the code and modifify from the code if youn need it
```bash
unzip COCO_subfolder_output.zip
```
First generate the TEM, take into account that the file **file_annotations/sal_ground_truth_emb_SALICON_TEM_w.txt** structures **file_annotations/sal_ground_truth_emb_ADE20K_all_image_co_occur.txt**. If you want to generate the embedding from scratch and first define your new training folder before run - refer to the code to see the specific details in **generate_objects_co_occur.py**:
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

Now you must create the **centerbias** files for the stimuli and the TEM images. For doing that be sure the stimuli_train.hdf5, fixations_train.hdf5, stimuli_val.hdf5, fixations_val.hdf5, stimuli_TEM_train.hdf5, and stimuli_TEM_val.hdf5 files are located in the **experiment_root/** folder. If you want to create your own sitmuli and fixations hdf5 files feel free to modify the **create_stimuli.py** and **create_fixations.py** files. Subsequently, you can run: 
```python
python create_centerbias.py
python create_centerbias_TEM.py
```
Now, you are ready to do the training! and you can run the following command. Check the configuration file .yaml, in this case the **config_dg2_TEM.yaml**:
```python
python run_dgII_evaluation_plus_TEM.py
```
While you run the training and the test, or after **run_dgII_evaluation_plus_TEM.py** is done executing, a log.txt files will present you the current status of the learning, or if you want to wait until the end of the training  the file **results_TEM.csv** will show you the final Log-likehood (LL), Information Gain (IG), Area Under Curve (AUC), and Normalized Scanpath Saliency (NSS).  

A LL evolution through the training epochs could be observed in the following Figure including TEM features. This is the full pipeline of our new approach **DeepGaze+TEM**. An [AdaBound](https://github.com/Luolc/AdaBound) optimizer and a final Drop-out layer (before the Finalizer) must be added to the network for avoiding overfitting. The full pipeline of out semantic-based gaze prediction is shown in the following Figure:

<img src="https://github.com/meiyor/DeepGaze-Text-Embedding-Map/blob/main/pipeline_def_new_no_scan.jpg" width="1100" height="300">

The performance comparison between the [DeepGaze (DG)](https://github.com/matthias-k/deepgaze_pytorch) baseline and outr DeepGaze+TEM approach is the following. Use the code on [**matlab_metrics**](https://github.com/meiyor/DeepGaze-Text-Embedding-Map/tree/main/matlab_metrics) directory to compute the based on the resulting .csv file after training and testing using **run_dgII_evaluation_plus_TEM.py**:

|   | **IG** | **LL** | **AUC** | **NSS** | 
| ------------- | ------------- |  ------------- | ------------- |  ------------- |
| **DeepGaze**  | 0.5414±0.5714 | 1.2296±0.5714 | 0.8317±0.0562 | 1.5268±0.7245 |
| **DeepGaze + TEM**  | **0.5662±0.5816** | **1.2556±0.5816** | 0.8333±0.0563 | **1.5445±0.7661** | 

A statistical comparison have been using paired tests, evaluating the normality of each metrics and finding the following effect plot - correcting p-values.
All metrics show a significant improvement for all the paired test as we shown in the following table. The methods on * are less affected by the sample normality, for instance the signrank tests.

<img src="https://github.com/meiyor/DeepGaze-Text-Embedding-Map/blob/main/plot_effect_IG.jpg" width="750" height="350">

|   | **IG** | **LL** | **NSS** | **AUC** | 
| ------------- | ------------- |  ------------- | ------------- |  ------------- |
| **ttest**  | 2.25E-12 | 2.25E-12 | 1.72E-5 | 5.41E-7 |
| **signtest***  | 5.56E-12 | 5.56E-12 | 1.33E-9 | 1.41E-5 | 
| **signrank***  | 3.03E-14 | 3.03E-14 | 2.94E-9 | 7.88E-8 | 

A great example of the metrics example using the TEM as features is shown on the next figure. This example describes the saliency map pattern using a jet colormap, thus showing the fixations groundtruth, the panoptic annotation image, and the results for the DG baseline and our proposed DG+TEM showing better results for the DG+TEM approach. The panoptic groundtruth image can be obtained from the COCO dataset or the panoptic/semantic scene segmentation networks proposed by the remarkable paper of [Zhou et., al 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_Scene_Parsing_Through_CVPR_2017_paper.html) 


<img src="https://github.com/meiyor/DeepGaze-Text-Embedding-Map/blob/main/example_saliency_pattern.jpg" width="850" height="500">

