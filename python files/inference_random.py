# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 16:26:01 2021

@author: Abhimanyu
"""


import os
import cv2
import random

# paths to files
path_to_test_folder = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/test/dbai_test_data/test_data/'
path_to_train_folder = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/train/dbai_train_data/dbai_data/train/'
path_to_class_file = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/test/dbai_test_data/obj.names'
cfg_file = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/test/dbai_test_data/yolov4.cfg'
weights_file = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/test/dbai_test_data/yolov4-obj_final.weights'
random_test = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/test/dbai_test_data/random test/'
random_train = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/test/dbai_test_data/random train/'

with open(path_to_class_file, 'r') as f:
    classes = f.read().splitlines()

test_imgs = os.listdir(path_to_test_folder)
train_imgs = []
ls= os.listdir(path_to_train_folder)
for file in ls:
    if file.endswith(".jpg"):
        train_imgs.append(file)

test_random = random.sample(test_imgs,10)
train_random = random.sample(train_imgs,10)

print(test_random)
print(train_random)

net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

for image in test_random:
    img = cv2.imread(path_to_test_folder+image)
    classIds, scores, boxes = model.detect(img, confThreshold=0.5, nmsThreshold=0.3)
    for (classId, score, box) in zip(classIds, scores, boxes):
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),color=(0, 255, 0), thickness=2)
 
        text = '%s: %.2f' % (classes[classId[0]], score)
        cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,color=(0, 255, 0), thickness=2)
    cv2.imwrite(random_test+image,img)
    
for image in train_random:
    img = cv2.imread(path_to_train_folder+image)
    classIds, scores, boxes = model.detect(img, confThreshold=0.5, nmsThreshold=0.3)
    for (classId, score, box) in zip(classIds, scores, boxes):
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),color=(0, 255, 0), thickness=2)
 
        text = '%s: %.2f' % (classes[classId[0]], score)
        cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,color=(0, 255, 0), thickness=2)
    cv2.imwrite(random_train+image,img)



