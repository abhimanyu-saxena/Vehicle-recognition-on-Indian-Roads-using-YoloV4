# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 01:24:13 2021

@author: Abhimanyu
"""

from shutil import copy, move
import random
import os

# define paths
datadir = "C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/train/dbai_train_data/dbai_data/"
img_dir = "C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/train/dbai_train_data/dbai_data/JPEGImages/"
train_path = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/train/dbai_train_data/dbai_data/train/'
val_path = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/train/dbai_train_data/dbai_data/val/'
no_obj_path = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/train/dbai_train_data/dbai_data/no label img/'

#get list of all files in dataset 
sorted_list_img = sorted(os.listdir(img_dir))

# define list
img_list = []
label_list = []
no_obj_list =[]
train = []
val=[]

# seperate the images and txt files in different lists
for i in sorted_list_img:
    if(str(os.path.splitext(i)[1]) == '.txt'):
        label_list.append(os.path.splitext(i)[0])
    elif (str(os.path.splitext(i)[1]) == '.jpg'):
        img_list.append(os.path.splitext(i)[0])

# check if image has corresponding text file
# if not move that image to no object folder
for j in img_list:
    if j not in label_list:
        no_obj_list.append(j)
        move(datadir+'/JPEGImages/'+j+'.jpg', no_obj_path)

# shuffle the label list to for val and train split
random.shuffle(label_list)
t_split = 0.8 # change the split for train and val
train = img_list[:int(t_split*len(img_list))+1]
val = img_list[-int(len(img_list)*(1-t_split)):]

# copy training images and labels to train folder and create ocj.data lists
for a in train:
    copy(datadir+'/JPEGImages/'+a+'.jpg', train_path)
    copy(datadir+'/JPEGImages/'+a+'.txt', train_path)
    f = open(datadir+'/train.txt', "a")
    i="data/obj/"+a+'.jpg'
    f.write(a)
    f.write('\n')
    f.close()
        
    
# copy val images and labels to val folder
for b in val:
    copy(datadir+'/JPEGImages/'+b+'.jpg', val_path)
    copy(datadir+'/JPEGImages/'+b+'.txt', val_path)
    f = open(datadir+'/test.txt', "a")
    i="data/test/"+b+'.jpg'
    f.write(b)
    f.write('\n')
    f.close()

        















