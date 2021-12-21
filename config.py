#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:50:38 2021

@author: thomas_yang
"""

import os

classes_label = {"person": 0}

finetune = True
finetune_load_epoch = 19

batch_size = 20
epochs = 100
learning_rate = 5e-4

get_image_size = (416, 416)
max_boxes_per_image = 100
downsampling_ratio = 4

num_classes = 1
tid_classes = 13385
# tid_classes = 55

heads = {"heatmap": num_classes, "wh": 2, "reg": 2, "embed": 256,"tid": tid_classes}
bif = 256

hm_weight = 1.0
off_weight = 1.0
wh_weight = 0.1
reid_eright = 0.1

train_data_dir = '/home/thomas_yang/ML/hTG_MOT_pytorch/txt_file/'
train_data_list = os.listdir(train_data_dir)
train_data_list.sort()
train_data_list = [train_data_dir + i for i in train_data_list]

top_K = 50
score_threshold = 0.3

