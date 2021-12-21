#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 12:10:35 2021

@author: thomas_yang
"""

import matplotlib.pyplot as plt
import pickle 
import config
import numpy as np
import os
import cv2
import torch
from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.utils import mkdir_if_missing
import datetime

if __name__ == '__main__':
 
    #%%   
    video_path = '/home/thomas_yang/Downloads/Viveland-records-20210422'
    video_1 = 'vlc-record-2021-04-22-15h48m40s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_2 = 'vlc-record-2021-04-22-16h00m58s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_3 = 'vlc-record-2021-04-22-16h05m31s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_4 = 'vlc-record-2021-04-22-16h12m47s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_5 = 'vlc-record-2021-04-22-16h23m28s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_6 = 'vlc-record-2021-04-22-16h31m33s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_7 = 'vlc-record-2021-04-22-17h02m58s-rtsp___192.168.102.3_8554_fhd-.mp4'

    # video_path = '/home/thomas_yang/ML/datasets/RaiseHand/vlc-record-2021-12-11-15h28m12s-rtsp___10.10.0.5_28554_fhd-'
    # video_1 = 'vlc-record-2021-12-11-15h23m44s-rtsp___10.10.0.5_28554_fhd-.mp4'
    # video_2 = 'vlc-record-2021-12-11-15h24m59s-rtsp___10.10.0.5_28554_fhd-.mp4'
    # video_3 = 'vlc-record-2021-12-11-15h28m12s-rtsp___10.10.0.5_28554_fhd-.mp4'
    
    # video_path = '/home/thomas_yang/ML/datasets/TestVideo'
    # video_1 = 'Street_1.mp4'  
    
    target_video = video_7
    show_image=True     
    use_cuda=True
    frame_rate=30 

    #%% Embed Part Tracker

    embed_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '_embed'
    save_dir='./tracker_outputs/' + embed_tag

        
    if save_dir:
        mkdir_if_missing(save_dir)
        
    tracker_embed = JDETracker(frame_rate=frame_rate, is_embed=True)
    timer = Timer()
    results = []
    frame_id = 0
    
    cap = cv2.VideoCapture(os.path.join(video_path, target_video))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if frame_id % frame_rate == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        img0 = np.copy(frame)
        img = cv2.resize(img0, config.get_image_size)[...,::-1]
        img = img / 255.
        img = img.transpose(2, 0, 1).astype(np.float32)    
        
        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)

        online_targets = tracker_embed.update(blob, img0)
        # print('online_targets', online_targets)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            min_box_area = 100
            if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        
        online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time)    
        cv2.imshow('online_im', cv2.resize(online_im, (960, 540)))
        cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1

        if cv2.waitKey(1) == ord('q'):
            break    
    cv2.destroyAllWindows()
    
    with open('/home/thomas_yang/ML/hTG_MOT_pytorch/tracker_outputs/%s_results.pkl' %embed_tag, 'wb') as f:
        pickle.dump(results, f)

    #%% IoU Part Tracker
    
    iou_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '_iou'
    save_dir='./tracker_outputs/' + iou_tag

        
    if save_dir:
        mkdir_if_missing(save_dir)
        
    tracker_iou = JDETracker(frame_rate=frame_rate, is_embed=False)
    timer = Timer()
    results = []
    frame_id = 0
    
    cap = cv2.VideoCapture(os.path.join(video_path, target_video))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if frame_id % frame_rate == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        img0 = np.copy(frame)
        img = cv2.resize(img0, config.get_image_size)[...,::-1]
        img = img / 255.
        img = img.transpose(2, 0, 1).astype(np.float32)    
        
        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)

        online_targets = tracker_iou.update(blob, img0)
        # print('online_targets', online_targets)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            min_box_area = 100
            if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        
        online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time)    
        cv2.imshow('online_im', cv2.resize(online_im, (960, 540)))
        cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
        

        if cv2.waitKey(1) == ord('q'):
            break    
    cv2.destroyAllWindows()
    
    with open('/home/thomas_yang/ML/hTG_MOT_pytorch/tracker_outputs/%s_results.pkl' %iou_tag, 'wb') as f:
        pickle.dump(results, f)    
    
#%% Plot Part

    # embed_tag = '20211217_140102_embed'
    # iou_tag = '20211217_141110_iou'

    data_dir = '/home/thomas_yang/ML/hTG_MOT_pytorch/tracker_outputs/'
    dir_a = data_dir + embed_tag
    dir_b = data_dir + iou_tag
    video_name = data_dir + '%s_%s_combine.avi' %(embed_tag, iou_tag)
    
    with open(data_dir + '%s_results.pkl' %embed_tag, 'rb') as f:
        pickle_a = pickle.load(f)
    with open(data_dir + '%s_results.pkl' %iou_tag, 'rb') as f:
        pickle_b = pickle.load(f)
    
    frame_id = [0]
    max_id_a = [0]
    max_id_b = [0]
    for idx, (a, b) in enumerate(zip(pickle_a, pickle_b)):
        frame_id.append(idx)
        max_id_a.append(max(np.max(a[2]), np.max(max_id_a))) if len(a[2]) > 0 else max_id_a.append(max_id_a[-1])
        max_id_b.append(max(np.max(b[2]), np.max(max_id_b))) if len(b[2]) > 0 else max_id_b.append(max_id_b[-1])
    
    fig = plt.figure(1)        
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(frame_id, max_id_a, linewidth=6, color=(0.12, 0.46, 0.70))
    plt.plot(frame_id, max_id_b, linewidth=4, color=(1, 0.49, 0.05))
    
    list_a_y = sum([a[2] for a in pickle_a], [])
    list_a_x =sum([[a[0]]*len(a[2]) for a in pickle_a], [])
    plt.plot(list_a_x, list_a_y, marker='_', linewidth=0, markerfacecolor='#9acd32', markeredgecolor='#9acd32', markersize = 4)
    
    list_b_y = sum([b[2] for b in pickle_b], [])
    list_b_x = sum([[b[0]]*len(b[2]) for b in pickle_b], [])
    plt.plot(list_b_x, list_b_y, marker='_', linewidth=0, markerfacecolor='#ba55d3', markeredgecolor='#ba55d3', markersize = 2)

    plt.legend(('Only Embedded', 'Only IoU', 'ReID'))
    plt.xlabel('Frame')
    plt.ylabel('Max ID Num')
    plt.title('Embedded vs IoU')
    major_ticks_x = np.arange(0, len(frame_id), 1000)
    minor_ticks_x = np.arange(0, len(frame_id), 100)
    major_ticks_y = np.arange(0, max(max_id_a[-1], max_id_b[-1]), 10)
    minor_ticks_y = np.arange(0, max(max_id_a[-1], max_id_b[-1]), 2)
    
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)    
    plt.grid(which='both')    
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    plt.legend(('Embedded maxima ID', 'IoU maxima ID', 'Embedded active ID', 'IoU active ID'))
    plt.show()
    plt.pause

#%% Combine Video

    list_a = os.listdir(dir_a)
    list_a.sort()
    list_a = [dir_a + '/' + i for i in list_a]
    
    list_b = os.listdir(dir_b)
    list_b.sort()
    list_b = [dir_b + '/' + i for i in list_b]
    
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
