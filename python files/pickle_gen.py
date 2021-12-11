# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 01:42:35 2021

@author: Abhimanyu
"""
# import libraries
import os
import pickle
import cv2

#Helper functions
def pickleIt(lst, path):
    with open(path+'/predicted_test.pickle', 'wb') as f:
        pickle.dump(lst,f)

# paths to files
path_to_data_folder = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/test/dbai_test_data/test_data/'
#path_to_data_folder = 'C:/Users/Abhimanyu/Desktop/deeplearningchallenge-master (1)/deeplearningchallenge-master/data/train/dbai_train_data/dbai_data/JPEGImages/'
path_to_class_file = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/test/dbai_test_data/obj.names'
cfg_file = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/test/dbai_test_data/yolov4.cfg'
weights_file = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/test/dbai_test_data/yolov4-obj_final.weights'
pickle_path = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/test/dbai_test_data'
#pickle_path = 'C:/Users/Abhimanyu/Desktop/deeplearningchallenge-master (1)/deeplearningchallenge-master/data/train/dbai_train_data/dbai_data/'

with open(path_to_class_file, 'r') as f:
    classes = f.read().splitlines()

pred = []

test_imgs = os.listdir(path_to_data_folder)

net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

for image in test_imgs:
    img = cv2.imread(path_to_data_folder+image)
    classIds, scores, boxes = model.detect(img, confThreshold=0.5, nmsThreshold=0.3)
    ls=[]
    image_data=[]
    for (classId, score, box) in zip(classIds, scores, boxes):
        a=[]
        bbox=[]
        x1,y1,x2,y2 = (box[0], box[1], box[0] + box[2], box[1] + box[3])
        bbox.extend((x1,y1,x2,y2))
        a.extend((int(classId), str(classes[classId[0]]),round(score[0],2)))
        a.append(bbox)
        ls.append(a)
    image_data.extend((image,ls))
    pred.append(image_data)

#print(len(pred))
pickleIt(pred,pickle_path)





