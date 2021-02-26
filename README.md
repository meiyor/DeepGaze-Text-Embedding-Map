## DeepGaze-Text-Embedding-Map 
This repository includes the implementation of **DeepGaze** adding the **Text-Embedding-Maps** (TEM) [Barman et., al 2020,Yang et., al 2017]for predicting robustly human fixation/gaze.

We use a data intersection between [COCO](https://cocodataset.org/#home) and [SALICON](https://cocodataset.org/#home) to perform our evaluations using the fixations from SALICON and the panoptic annotations/segmentations from COCO.

First download the data from the Onedrive .zip file:

[https://drive.google.com/uc?export=download&id=1RM4gXlSIic22HvYHaS5XOGmjcLDSUiUv](https://drive.google.com/uc?export=download&id=1RM4gXlSIic22HvYHaS5XOGmjcLDSUiUv)

For runnig the code first generate the TEM:


```python
python generate_TEM.py
```
