# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 23:52:04 2021

@author: Abhimanyu
"""


import json
from matplotlib import pyplot as plt

# replace with path to json file
f= open('C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/train/dbai_train_data/dbai_data/annotations.json',)
training_data = json.load(f)


classes = ['animal','autorickshaw','bicycle','bus','car','motorbike','person','rider','tempo','truck']

distr = [0]*10

# get count of object from annotations
for i in range(len(training_data['annotations'])):
    image_id = (training_data['annotations'][i]['image_id'])
    category_id = (training_data['annotations'][i]['category_id'])
    distr[category_id] = distr[category_id]+1


obj = classes
count = distr

# Figure Size
fig, ax = plt.subplots(figsize =(10, 5))
 
# Horizontal Bar Plot
ax.barh(obj, count)
 
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
 
# Remove x, y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
 
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)
 
# Add x, y gridlines
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
 
# Show top values
ax.invert_yaxis()
 
# Add annotation to bars
for i in ax.patches:
    if round((i.get_width()), 2)< 100:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                 str(round((i.get_width()), 2)),
                 fontsize = 10, fontweight ='bold',
                 color ='red')
        
    elif round((i.get_width()), 2)< 500:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                 str(round((i.get_width()), 2)),
                 fontsize = 10, fontweight ='bold',
                 color ='orange')
    else:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                 str(round((i.get_width()), 2)),
                 fontsize = 10, fontweight ='bold',
                 color ='green')
 
# Add Plot Title
ax.set_title('Instances of objects in dataset',
             loc ='left', )
 
# Add Text to chart
fig.text(0.9, 0.15, 'Total: '+str(sum(distr)), fontsize = 12,
         color ='black', ha ='right', va ='bottom',
         alpha = 1)
 
# save Plot
plt.savefig('C:/Users/Abhimanyu/Downloads/deeplearningchallenge-master/data/train/dbai_train_data/dbai_data/foo.png')