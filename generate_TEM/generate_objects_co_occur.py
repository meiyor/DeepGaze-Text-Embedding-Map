from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import numpy as np
import json

import csv
import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from panopticapi.utils import IdGenerator, rgb2id

## use the general file with the annotation and we can query the images IDs
json_file = 'panoptic_annotation/panoptic_train2017.json'
category_file = 'panoptic_coco_categories.json'
directory_images_train='COCO/train/'
directory_images_val='COCO/val/'
csv_file='COCO_SALICON_train_only_def_annotation.csv'

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
       #print(row,'n')
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
#print(coco_d['annotations'],'anno_vale',coco_d['annotations'][17],'anno_a1',category_res[0]['id'])

from os import listdir
from os.path import isfile, join

onlyfiles_train = [f1 for f1 in listdir(directory_images_train) if isfile(join(directory_images_train, f1))]
onlyfiles_val = [f2 for f2 in listdir(directory_images_val) if isfile(join(directory_images_val, f2))]

print(category_res)
## matrix construction

## the co-occurrence matrix is 116 to 116 objects from this specific data intersection
co_occur_mat=np.zeros((117,117))
object_list=[]
object_list_scenes=['kitchen','bedroom','living_room','bathroom','corridor','dinning_room','office']
ind=0
for file_name in onlyfiles_train:
    f_name=file_name.split('_')
    ff_name=f_name[2].split('.')
    file_name_def=ff_name[0]
    ## check the class  and the name for scene
    for g in range(0,len(name_def)):
        #print(file_name_def,name_def)
        if file_name_def==name_def[g]:
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
           ## if this other scene it is described like 7 ind_val you can add more if you decide, in COCO is more complicated because it is not an scene but a supercategory and in most times is part of the objects     
           else:
              ind_val=7
           break
    for k in range(0,len(annotation_object)):
       file_annotated=annotation_object[k]['file_name'].split('.')
       if file_annotated[0] == file_name_def:
           #print(file_name_def,'name')
           segmentation=annotation_object[k]['segments_info']
           num_objects=[]
           for i in range(0,len(segmentation)):
               id_inx=segmentation[i]['category_id']
               #print(segmentation[i],id_inx,len(category_res))
               ## look for category number
               for c in range(0,len(category_res)):
                   if category_res[c]['id'] == id_inx:
                      name_object=category_res[c]['name']
                      print(category_res[c])
                      break
               if not(name_object in object_list):
                   object_list.append(name_object)
                   if ind>=1 and len(num_objects):
                      for p in range(0,len(num_objects)):
                          #print(ind,num_objects,num_objects[p],'kk')
                          co_occur_mat[ind+7,num_objects[p]+7]=co_occur_mat[ind+7,num_objects[p]+7]+1
                          print(ind_val,'ind_val')
                          if ind_val<=6:
                             co_occur_mat[ind_val,num_objects[p]+7]=co_occur_mat[ind_val,num_objects[p]+7]+1
                      #num_objects.append(ind[0])
                   #else:
                   num_objects.append(ind)
                   ind=ind+1
                   #print(ind,'eee')
               else:
                     ind_n = [object_list.index(i_n) for i_n in object_list  if name_object in i_n]
                     num_objects.append(ind_n[0])
                     if not(ind_n[0]+7 == ind+7):
                        for p in range(0,len(num_objects)):
                          if not(ind_n[0]+7 == num_objects[p]+7):
                              #print(ind_val,'ind_val')
                              #print(ind_n,num_objects[p],'ll')
                              co_occur_mat[ind_n[0]+7,num_objects[p]+7]=co_occur_mat[ind_n[0]+7,num_objects[p]+7]+1
                              if ind_val<=6:
                                 co_occur_mat[ind_val,num_objects[p]+7]=co_occur_mat[ind_val,num_objects[p]+7]+1 

           break
print(co_occur_mat,object_list,len(object_list))
np.savetxt('sal_cooccur_mat_new.txt',co_occur_mat,delimiter=',')
object_list=object_list_scenes+object_list
with open('index_objects_new','w') as result_file:
    wr = csv.writer(result_file)
    wr.writerow(object_list)
#np.savetxt('index_objects',object_list,delimiter=',')
#print(onlyfiles_train,'train')

