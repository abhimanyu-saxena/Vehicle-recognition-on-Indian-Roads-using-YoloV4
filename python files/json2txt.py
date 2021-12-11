# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 01:24:13 2021

@author: Abhimanyu
"""

import cv2
import json
import os
f= open('C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/train/dbai_train_data/dbai_data/annotations.json',)
hdr = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/train/dbai_train_data/dbai_data/'
training_data = json.load(f)

def sorting(l1, l2):
    if l1 > l2:
        lmax, lmin = l1, l2
        return lmax, lmin
    else:
        lmax, lmin = l2, l1
        return lmax, lmin
def get_img_shape(path):
    img = cv2.imread(path)
    try:
        return img.shape
    except AttributeError:
        print("error! ", path)
        return (None, None, None)
def convert_labels(path, x1, y1, x2, y2):

    
    
    size = get_img_shape(path)
    xmax, xmin = sorting(x1, x2)
    ymax, ymin = sorting(y1, y2)
    dw = 1./size[1]
    dh = 1./size[0]
    x = (xmin + xmax)/2.0
    y = (ymin + ymax)/2.0
    w = xmax - xmin
    h = ymax - ymin
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
check_set = set()


for i in range(len(training_data['annotations'])):
    image_id = (training_data['annotations'][i]['image_id'])
    category_id = str(training_data['annotations'][i]['category_id'])
    bbox = training_data['annotations'][i]['bbox']
    image_path = hdr+training_data['images'][image_id]['file_name']
    #print(image_path)
    kitti_bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
    yolo_bbox = convert_labels(image_path, kitti_bbox[0], kitti_bbox[1], kitti_bbox[2], kitti_bbox[3])
    filename = hdr + str(os.path.splitext(training_data['images'][image_id]['file_name'])[0]) + ".txt"
    #print(filename)
    content = category_id + " " + str(yolo_bbox[0]) + " " + str(yolo_bbox[1]) + " " + str(yolo_bbox[2]) + " " + str(yolo_bbox[3])
    if image_id in check_set:
        # Append to file files
        file = open(filename, "a")
        file.write("\n")
        file.write(content)
        file.close()
    elif image_id not in check_set:
        check_set.add(image_id)
        # Write files
        file = open(filename, "w+")
        file.write(content)
        file.close()

















