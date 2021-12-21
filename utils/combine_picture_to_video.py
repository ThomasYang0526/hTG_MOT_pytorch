#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 11:47:03 2021

@author: thomas_yang
"""

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

data_dir = '/home/thomas_yang/ML/hTG_MOT_pytorch/tracker_outputs/'
dir_a = data_dir + '20211216-151750_embed/'
dir_b = data_dir + '20211216-152113_iou/'
video_name = data_dir + 'combine.avi'

with open(data_dir + '20211216-151750_embed-results.pkl', 'rb') as f:
    pickle_a = pickle.load(f)
with open(data_dir + '20211216-152113_iou-results.pkl', 'rb') as f:
    pickle_b = pickle.load(f)

frame_id = [0]
max_id_a = [0]
max_id_b = [0]
for idx, (a, b) in enumerate(zip(pickle_a, pickle_b)):
    frame_id.append(idx)
    max_id_a.append(max(np.max(a[2]), np.max(max_id_a))) if len(a[2]) > 0 else max_id_a.append(max_id_a[-1])
    max_id_b.append(max(np.max(b[2]), np.max(max_id_b))) if len(b[2]) > 0 else max_id_b.append(max_id_b[-1])
       
plt.figure(1)
plt.plot(frame_id, max_id_a, linewidth=4, color=(0.12, 0.46, 0.70))
plt.plot(frame_id, max_id_b, linewidth=2, color=(1, 0.49, 0.05))
plt.legend(('Only Embedded', 'Only IoU', 'ReID'))
plt.xlabel('Frame')
plt.ylabel('Max ID Num')
plt.title('Vive Land Tracking Embedded vs IoU')
plt.grid(True)

t1=[a[2] for a in pickle_a]
t2=[[a[0]]*len(a[2]) for a in pickle_a]
t1 = sum(t1, [])
t2 = sum(t2, [])
plt.plot(t2, t1, marker='*', linewidth=0, markerfacecolor=(0.12, 0.46, 0.70), markeredgecolor=(0.12, 0.46, 0.70), markersize = 4)

t1=[b[2] for b in pickle_b]
t2=[[b[0]]*len(b[2]) for b in pickle_b]
t1 = sum(t1, [])
t2 = sum(t2, [])
plt.plot(t2, t1, marker='x', linewidth=0, markerfacecolor=(1, 0.49, 0.05), markeredgecolor=(1, 0.49, 0.05), markersize = 5)

# for idx, a in enumerate(pickle_a):
#     plt.plot([idx]*len(a[2]), a[2], marker='>', linewidth=0, markerfacecolor=(0.12, 0.46, 0.70), markeredgecolor=(0.12, 0.46, 0.70), markersize = 4)

# for idx, b in enumerate(pickle_b):
#     plt.plot([idx]*len(b[2]), b[2], marker='+', linewidth=0, markerfacecolor=(1, 0.49, 0.05), markeredgecolor=(1, 0.49, 0.05), markersize = 2)

plt.legend(('Embedded maxima ID', 'IoU maxima ID', 'Embedded active ID', 'IoU active ID'))
plt.pause

#%%

list_a = os.listdir(dir_a)
list_a.sort()
list_a = [dir_a + i for i in list_a]

list_b = os.listdir(dir_b)
list_b.sort()
list_b = [dir_b + i for i in list_b]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_name, fourcc, 30.0, (1920, 1080*2))
for idx, (a, b) in enumerate(zip(list_a, list_b)):
    print(idx , len(pickle_a))
    img_a = cv2.imread(a)
    img_b = cv2.imread(b)
    combine = cv2.vconcat((img_a, img_b))    
    
    if not pickle_a[idx][2] == pickle_b[idx][2]:
        cv2.rectangle(combine, (20, 20), (combine.shape[1]-20, combine.shape[0]-20), (0, 0, 255), 20)
    
    out.write(combine)
    # cv2.imshow('com', cv2.resize(combine, (960, 1080)))    
    # if cv2.waitKey(1) == ord('q'):
    #     break

cv2.destroyAllWindows()                          
out.release()
    

    
