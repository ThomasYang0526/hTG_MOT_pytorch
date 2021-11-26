#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 19:10:34 2021

@author: thomas_yang
"""

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
 
    # result_filename 
    save_dir='./tracker_outputs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    show_image=True 
    frame_rate=30 
    use_cuda=True
        
    if save_dir:
        mkdir_if_missing(save_dir)
        
    tracker = JDETracker(frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    
    video_path = '/home/thomas_yang/Downloads/Viveland-records-20210422'
    video_1 = 'vlc-record-2021-04-22-15h48m40s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_2 = 'vlc-record-2021-04-22-16h00m58s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_3 = 'vlc-record-2021-04-22-16h05m31s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_4 = 'vlc-record-2021-04-22-16h12m47s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_5 = 'vlc-record-2021-04-22-16h23m28s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_6 = 'vlc-record-2021-04-22-16h31m33s-rtsp___192.168.102.3_8554_fhd-.mp4'
    video_7 = 'vlc-record-2021-04-22-17h02m58s-rtsp___192.168.102.3_8554_fhd-.mp4'
    
    cap = cv2.VideoCapture(os.path.join(video_path, video_7))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # half_point = length//8.8
    # cap.set(cv2.CAP_PROP_POS_FRAMES, half_point)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        img0 = np.copy(frame)
        img = cv2.resize(img0, config.get_image_size)[...,::-1]
        img = img / 255.
        img = img.transpose(2, 0, 1).astype(np.float32)    
        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            min_box_area = 100
            if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    import pickle 
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
 
    with open('results.pkl', 'rb') as f:
        a = pickle.load(f)
    


