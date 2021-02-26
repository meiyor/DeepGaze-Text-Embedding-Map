from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import numpy as np
import json
#import scipy.io
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

import csv
import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from tifffile import imread, imwrite
from panopticapi.utils import IdGenerator, rgb2id

## use the general file with the annotation and we can query the images IDs
json_file = 'panoptic_train2017.json'
img_folder='panoptic_train2017'
category_file = 'panoptic_coco_categories.json'
directory_images_train='COCO/train/'
directory_images_val='COCO/val/'
csv_file='COCO_SALICON_train_only_def_annotation.csv'
output_dir='TEM_output_all_train_dist_new'

## read the semantic results files from the images itself
name_vectors=[]
vector_data=[]
data_vectors='sal_ground_truth_emb_SALICON_TEM_w.txt'
with open(data_vectors,newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            name_vectors.append(row[0])
            val_float=[float(i) for i in row[1:301]]
            vector_data.append(val_float)

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X = vector_data)

pca_data=pca

name_def=[]
category_def=[]
## read the annotation csv file a-priori
with open(csv_file, mode='r') as csv_file_f:
    csv_reader = csv.DictReader(csv_file_f)
    line_count = 0
    for row in csv_reader:
       image_name=row['name'].split('_')
       name_im=image_name[2].split('.')
       name_def.append(name_im[0])
       category_def.append(row['Category'])
       line_count=line_count+1

## read the whole json file
with open(json_file, 'r') as f:
    coco_d = json.load(f)

## read the panoptic category file
with open(category_file, 'r') as f_c:
    coco_category = json.load(f_c)


annotation_object=coco_d['annotations']
category_res=coco_category

for c in range(0,len(category_res)):
    print(category_res[c]['id'],c)

from os import listdir
from os.path import isfile, join

onlyfiles_train = [f1 for f1 in listdir(directory_images_train) if isfile(join(directory_images_train, f1))]
onlyfiles_val = [f2 for f2 in listdir(directory_images_val) if isfile(join(directory_images_val, f2))]

## with the file lists we will create a TEM image with the distance between the object annotation and the scene corresponding to each file

object_list=[]
object_list_scenes=['kitchen','bedroom','living_room','bathroom','corridor','dinning_room','office']
ind=0
cc=0
for file_name in onlyfiles_train:
    print(file_name,cc)
    f_name=file_name.split('_')
    ff_name=f_name[2].split('.')
    file_name_def=ff_name[0]
    ## check the class  and the name for scene
    for g in range(0,len(name_def)):
        #print(file_name_def,name_def)
        if file_name_def==name_def[g]:
           #print(cat_val)
           cat_val=category_def[g]
           if cat_val=='kitchen':
              ind_val=0
           elif cat_val=='bedroom':
              ind_val=1
           elif cat_val=='living_room':
              ind_val=2
           elif cat_val=='bathroom':
              ind_val=3
           elif cat_val=='corridor':
              ind_val=4
           elif cat_val=='dinning_room':
              ind_val=5
           elif cat_val=='office':
              ind_val=6
           else:
              ind_val=7
           break
    for k in range(0,len(annotation_object)):
       file_annotated=annotation_object[k]['file_name'].split('.')
       segment_burst=annotation_object[k]['segments_info']
       if file_annotated[0] == file_name_def:
           img = np.array(Image.open(os.path.join(img_folder,file_annotated[0]+'.png')))
           img_real = np.array(Image.open(os.path.join(directory_images_train,file_name)))
           img_sal = np.zeros((img.shape[0],img.shape[1],3))
           img_comp= np.zeros((img.shape[0],img.shape[1],3)) 
           segmentation_img=rgb2id(np.array(img,dtype=np.uint8))
           boundaries_img = find_boundaries(segmentation_img, mode='thick')
           for i in range(0,len(segment_burst)):
               id_inx=segment_burst[i]['id']
               id_name=segment_burst[i]['category_id']
               mask = segmentation_img == id_inx
               ## look for object name before matching that name with the embedding file
               for c in range(0,len(category_res)):
                   if category_res[c]['id'] == id_name:
                      name_object=category_res[c]['name']
                      break
               ## calculate cosine distance between objects and scene
               for q in range(0,len(name_vectors)):
                   if name_vectors[q]==name_object:
                     pos_n=q
                     break
               val_dist = cdist(np.reshape(vector_data[ind_val],(-1,300)),np.reshape(vector_data[pos_n],(-1,300)),'cosine')
               val_dist_1 = cdist(np.reshape(vector_data[ind_val],(-1,300)),np.reshape(vector_data[pos_n],(-1,300)),'euclidean')
               val_dist_2 = cdist(np.reshape(vector_data[ind_val],(-1,300)),np.reshape(vector_data[pos_n],(-1,300)),'chebyshev')
               img_sal[mask]=[val_dist[0][0],val_dist_1[0][0],val_dist_2[0][0]]
           img_sal[boundaries_img] = np.zeros((3))
           break
    print('COCO_train2014_'+file_annotated[0]+'.tif',cc)
    cc=cc+1
    imwrite(os.path.join(output_dir,'COCO_train2014_'+file_annotated[0]+'.tif'),img_sal,compress=6)

