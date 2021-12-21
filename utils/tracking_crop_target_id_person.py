#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:05:07 2021

@author: thomas_yang
"""

#%%
import os
import cv2

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('/home/thomas_yang/ML/hTG_MOT_pytorch/tracking.avi', fourcc, 20.0, (960, 540))
# imgae_dir = '/home/thomas_yang/ML/hTG_MOT_pytorch/tracker_outputs/20211125-125819/'
# image_lists = os.listdir(imgae_dir)
# image_lists.sort()
# image_lists = [imgae_dir + i for i in image_lists]

# for idx,  img_name in enumerate(image_lists):    
#     print(idx)               
#     image_array0 = cv2.imread(img_name)
#     image_array0 = cv2.resize(image_array0, (960, 540))

#     out.write(image_array0)
#     cv2.imshow("image_array0", image_array0)
#     if cv2.waitKey(1) == ord('q'):
#         break
    
# # cap.release()
# out.release()
# cv2.destroyAllWindows()

def letterbox(img, height=256, width=128,
              color=(127, 127, 127)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh

#%% Crop Target tid
import os
import cv2
import pickle
import sys 
import numpy as np

target_paerson_tid = 2   
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/thomas_yang/ML/hTG_MOT_pytorch/tracking_tid_%d.avi' %target_paerson_tid, fourcc, 30.0, (128, 256))
imgae_dir = '/home/thomas_yang/ML/hTG_MOT_pytorch/tracker_outputs/20211125-125819/'
image_lists = os.listdir(imgae_dir)
image_lists.sort()
image_lists = [imgae_dir + i for i in image_lists]
 
with open('/home/thomas_yang/ML/hTG_MOT_pytorch/tracker_outputs/20211125-125819-results.pkl', 'rb') as f:
    results = pickle.load(f)
 
for i_idx, i in enumerate(results):
    print(i_idx)
    if target_paerson_tid in i[2]:
        for j_idx, j in enumerate(i[2]):            
            if j == target_paerson_tid:                
                bbox = np.array(i[1][j_idx])
                bbox = bbox.astype(np.int)
                x, y, w, h = bbox
                img = cv2.imread(image_lists[i[0]-1])
                if x < 0: x = 0
                if y < 0: y = 0
                if x + w > img.shape[1]: w = img.shape[1] - x
                if y + h > img.shape[0]: h = img.shape[1] - y
                crop = img[y:y+h, x:x+w, :]
                crop, _, _, _ = letterbox(crop)
                cv2.imshow("crop", crop)
                out.write(crop)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    out.release()
                    sys.exit()





