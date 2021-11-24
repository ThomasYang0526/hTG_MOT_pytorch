#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:08:21 2021

@author: thomas_yang
"""

import torch
import cv2
import numpy as np
from decoder import Decoder
from resnetFPN import MyResNet50
from draw import draw_boxes_joint_on_image
import config

device = torch.device('cuda')
model = MyResNet50().cuda()
model.load_state_dict(torch.load('./saved_model/resnet50FPN_256_epoch_{}.pth'.format(8)))
model.eval()
on_video = True

#%%
if on_video:
    
    # from configuration import Config
    # import collections
    import os

    video_path = '/home/thomas_yang/ML/CenterNet_TensorFlow2/data/datasets/MPII'
    video_1 = 'mpii_512x512.avi'

    # video_path = '/home/thomas_yang/Downloads/2021-09-03-kaohsiung-5g-base-vlc-record'
    # video_1 = 'vlc-record-2021-09-03-12h47m06s-rtsp___10.10.0.37_28554_fhd-.mp4'
    # video_2 = 'vlc-record-2021-09-03-13h13m49s-rtsp___10.10.0.38_18554_fhd-.mp4' 
    # video_3 = 'vlc-record-2021-09-03-13h17m52s-rtsp___10.10.0.37_28554_fhd-.mp4'
    # video_4 = 'vlc-record-2021-09-03-13h23m50s-rtsp___10.10.0.25_18554_fhd-.mp4'
    
    video_path = '/home/thomas_yang/Downloads/Viveland-records-20210422'
    video_1 = 'vlc-record-2021-04-22-15h48m40s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_2 = 'vlc-record-2021-04-22-16h00m58s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_3 = 'vlc-record-2021-04-22-16h05m31s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_4 = 'vlc-record-2021-04-22-16h12m47s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_5 = 'vlc-record-2021-04-22-16h23m28s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_6 = 'vlc-record-2021-04-22-16h31m33s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_7 = 'vlc-record-2021-04-22-17h02m58s-rtsp___192.168.102.3_8554_fhd-.mp4'
    
    cap = cv2.VideoCapture(os.path.join(video_path, video_2))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half_point = length//7*1
    cap.set(cv2.CAP_PROP_POS_FRAMES, half_point)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('bbox_joint_01.avi', fourcc, 20.0, (2000,  540))
    
    embed_0 = np.zeros((540, config.heads["embed"]), dtype=np.float32)
    embed_1 = np.zeros((540, config.heads["embed"]), dtype=np.float32)
    inter = (np.ones((540,16,3))*(0,128,128)).astype(np.uint8)
    count = 0          
    xy0 = np.zeros((1,2))
    xy1 = np.zeros((1,2))
    while cap.isOpened():
        
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
                       
        image_array0 = np.copy(frame)
        img_r = cv2.resize(frame, config.get_image_size)[...,::-1]
        img_n = img_r / 255.
        img_n = np.expand_dims(img_n, 0)
        img_n = img_n.transpose(0,3,1,2)
        img_t = torch.from_numpy(img_n).cuda().type(torch.float32)
        
        with torch.no_grad():
            pred = model(img_t, tid = torch.ones((1, 100)).cuda())
            
        bboxes, scores, clses, embeds = Decoder(pred, (image_array0.shape[0], image_array0.shape[1]))
        argidx = np.argsort(bboxes[:, 0])

        xy0 = (bboxes[argidx[-2]][2:] + bboxes[argidx[-2]][0:2])//2
        xy1 = (bboxes[argidx[-1]][2:] + bboxes[argidx[-1]][0:2])//2
                  
        diff0 = np.copy(xy0[0])
        diff1 = np.copy(xy1[0])
        if diff0 < diff1:
            embed_0[count] = embeds[argidx[-2]].transpose()
            embed_1[count] = embeds[argidx[-1]].transpose()
            cv2.circle(img=image_array0, center=(int(xy0[0]), int(xy0[1])), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.circle(img=image_array0, center=(int(xy1[0]), int(xy1[1])), radius=5, color=(0, 255, 0), thickness=-1)
        else :
            embed_0[count] = embeds[argidx[-1]].transpose()
            embed_1[count] = embeds[argidx[-2]].transpose()
            cv2.circle(img=image_array0, center=(int(xy1[0]), int(xy1[1])), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.circle(img=image_array0, center=(int(xy0[0]), int(xy0[1])), radius=5, color=(0, 255, 0), thickness=-1)        

        
        # print(scores, clses, count, diff0, diff1)
        count += 1 
        
        image_with_boxes_joint_location_m = draw_boxes_joint_on_image(image_array0, bboxes, scores, clses)
        image_with_boxes_joint_location_m = cv2.resize(image_with_boxes_joint_location_m, (960, 540))
        image_with_boxes_joint_location_m = cv2.hconcat([image_with_boxes_joint_location_m, 
                                                         cv2.cvtColor((embed_0/100*255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                                                         inter,
                                                         cv2.cvtColor((embed_1/100*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)])
        
        out.write(image_with_boxes_joint_location_m)
        cv2.imshow("detect result", image_with_boxes_joint_location_m)
        if cv2.waitKey(1) == ord('q'):
            break
        
        if count == 420:
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()

#%%
    import matplotlib.pyplot as plt
    diff_00 = []
    diff_01 = []
    diff_10 = []
    diff_11 = []
    # for i in range(2, embed_0.shape[0]):
    for i in range(2, 419):
        diff_00.append(np.sqrt(np.sum((embed_0[i, :] - embed_0[i-1, :])**2))/embed_0.shape[1])
        diff_01.append(np.sqrt(np.sum((embed_0[i, :] - embed_1[i-1, :])**2))/embed_0.shape[1])
        diff_10.append(np.sqrt(np.sum((embed_1[i, :] - embed_0[i-1, :])**2))/embed_0.shape[1])
        diff_11.append(np.sqrt(np.sum((embed_1[i, :] - embed_1[i-1, :])**2))/embed_0.shape[1])
        
    diff_00 = np.array(diff_00)
    diff_01 = np.array(diff_01)
    diff_10 = np.array(diff_10)
    diff_11 = np.array(diff_11)
    
    x = np.arange(0, len(diff_00))
    plt.figure(0)
    plt.plot(x, diff_00, marker="o")
    plt.plot(x, diff_10, marker="o")
    plt.xlabel("Time Frame")
    plt.ylabel("Difference")
    plt.legend()
    plt.show()

    plt.figure(1)
    plt.plot(x, diff_01, marker="o")
    plt.plot(x, diff_11, marker="o")
    plt.xlabel("Time Frame")
    plt.ylabel("Difference")
    plt.legend()
    plt.show()
    
#%%
if on_video == False:
    img = cv2.imread('/home/thomas_yang/ML/datasets/MOT16/images/train/MOT16-02/img1/000001.jpg')
    img_r = cv2.resize(img, config.get_image_size)[...,::-1]
    img_n = img_r / 255.
    img_n = np.expand_dims(img_n, 0)
    img_n = img_n.transpose(0,3,1,2)
    img_t = torch.from_numpy(img_n).cuda().type(torch.float32)
    
    pred = model(img_t, tid = torch.ones((1, 100)).cuda())
    bboxes, scores, clses, embed_tmp = Decoder(pred, (img.shape[0], img.shape[1]))
    detect_img = draw_boxes_joint_on_image(img, bboxes, scores, clses)
        
    cv2.imshow('detect', detect_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




