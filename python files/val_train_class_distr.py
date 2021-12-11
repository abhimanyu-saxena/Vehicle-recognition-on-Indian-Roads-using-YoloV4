# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 08:55:48 2021

@author: Abhimanyu
"""

import os
# change paths to folders
train_path = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/train/dbai_train_data/dbai_data/train/'
val_path = 'C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/train/dbai_train_data/dbai_data/val/'

sorted_list_train = sorted(os.listdir(train_path))
sorted_list_val = sorted(os.listdir(val_path))

classes = ['animal','autorickshaw','bicycle','bus','car','motorbike','person','rider','tempo','truck']
count_train = [0]*10
count_val = [0]*10

for file in sorted_list_train:
    if file.endswith(".txt"):
        the_file = open(train_path+file, 'r')
        all_lines = the_file.readlines()
        for each_line in all_lines:
            classes = int(each_line[0])
            count_train[classes] = count_train[classes]+1
        the_file.close()

for file in sorted_list_val:
    if file.endswith(".txt"):
        the_file = open(val_path+file, 'r')
        all_lines = the_file.readlines()
        for each_line in all_lines:
            classes = int(each_line[0])
            count_val[classes] = count_val[classes]+1
        the_file.close()

distr_train = [0]*10
distr_val = [0]*10

for i in range(len(count_train)):
    distr_train[i] = round(count_train[i]/sum(count_train),3)

for i in range(len(count_val)):
    distr_val[i] = round(count_val[i]/sum(count_val),3)


print(classes)
print('Train distribution:\n',distr_train)
print('Validation distribution:\n',distr_val)


































