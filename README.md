## DeepGaze-Text-Embedding-Map 
This repository includes the implementation of **DeepGaze** adding the **Text-Embedding-Maps** (TEM) [Barman et., al 2020,Yang et., al 2017]for predicting robustly human fixation/gaze.

We use a data intersection between [COCO](https://cocodataset.org/#home) and [SALICON](https://cocodataset.org/#home) to perform our evaluations using the fixations from SALICON and the panoptic annotations/segmentations from COCO.

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
python generate_objects_co_occur.py
```
This will generate a new co-occurrences matrix in a file called **sal_cooccur_mat_new.txt**. To obtain the new embeddings use the instructions on the package **[Mittens](https://github.com/roamanalytics/mittens)** and upload the file **sal_cooccur_mat_new.txt** as csv.

```python
python generate_TEM.py
```
