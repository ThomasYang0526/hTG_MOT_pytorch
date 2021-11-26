#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 18:51:14 2021

@author: thomas_yang
"""

import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import math

import config
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
from torch.nn import Conv2d, Upsample, BatchNorm2d, ReLU, Softmax

class MyResNet50(ResNet):
    def __init__(self):
        super(MyResNet50, self).__init__(BasicBlock, [3, 4, 6, 3])
        self.conv_up1 = Conv2d( 64, config.bif, 1)
        self.conv_up2 = Conv2d(128, config.bif, 1)
        self.conv_up3 = Conv2d(256, config.bif, 1)
        self.conv_up4 = Conv2d(512, config.bif, 1)
        self.Upsample = Upsample(scale_factor=2)
        
        self.conv_heatmap_1 = Conv2d(config.bif, config.bif, 3, padding=(1, 1), bias=False)
        self.bn_heatmap = BatchNorm2d(config.bif)
        self.relu_heatmap = ReLU()
        self.conv_heatmap_2 = Conv2d(config.bif, config.heads["heatmap"], 1, padding=(0, 0), bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv_reg_1 = Conv2d(config.bif, config.bif, 3, padding=(1, 1), bias=False)
        self.bn_reg = BatchNorm2d(config.bif)
        self.relu_reg = ReLU()
        self.conv_reg_2 = Conv2d(config.bif, config.heads["reg"], 1, padding=(0, 0), bias=False)

        self.conv_wh_1 = Conv2d(config.bif, config.bif, 3, padding=(1, 1), bias=False)
        self.bn_wh = BatchNorm2d(config.bif)
        self.relu_wh = ReLU()
        self.conv_wh_2 = Conv2d(config.bif, config.heads["wh"], 1, padding=(0, 0), bias=False)

        self.conv_embed_1 = Conv2d(config.bif, config.bif, 3, padding=(1, 1), bias=False)
        self.bn_embed = BatchNorm2d(config.bif)
        self.relu_embed = ReLU()
        self.conv_embed_2 = Conv2d(config.bif, config.heads["embed"], 1, padding=(0, 0), bias=False)
        
        self.conv_reID = Conv2d(config.bif, config.tid_classes, 1, padding=(0, 0), bias=False)
        self.softmax = Softmax(dim=1)
        
    def forward(self, x, tid):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        p2_output = self.conv_up1(x1)
        p3_output = self.conv_up2(x2)
        p4_output = self.conv_up3(x3)
        p5_output = self.conv_up4(x4)
               
        p4_output = p4_output + self.Upsample(p5_output)
        p3_output = p3_output + self.Upsample(p4_output)
        p2_output = p2_output + self.Upsample(p3_output)

        heatmap = self.conv_heatmap_1(p2_output)
        heatmap = self.bn_heatmap(heatmap)
        heatmap = self.relu_heatmap(heatmap)
        heatmap = self.conv_heatmap_2(heatmap)

        reg = self.conv_reg_1(p2_output)
        reg = self.bn_reg(reg)
        reg = self.relu_reg(reg)
        reg = self.conv_reg_2(reg)        

        wh = self.conv_wh_1(p2_output)
        wh = self.bn_wh(wh)
        wh = self.relu_wh(wh)
        wh = self.conv_wh_2(wh)   

        tid_embed = self.conv_embed_1(p2_output)
        tid_embed = self.bn_embed(tid_embed)
        tid_embed = self.relu_embed(tid_embed)
        tid_embed = self.conv_embed_2(tid_embed)   
        
        tmp = torch.permute(tid_embed, (0, 2, 3, 1))
        tmp = torch.reshape(tmp, shape=(tmp.shape[0], -1, tmp.shape[-1]))
        idx = tid.type(torch.int64)
        idx = idx.unsqueeze(2)
        idx = idx.expand(idx.shape[0], config.max_boxes_per_image, config.heads["embed"])
        tmp = torch.gather(tmp, 1, idx)        
        tmp = tmp.unsqueeze(1)
        tmp = torch.permute(tmp, (0, 3, 1, 2))
        
        tid = self.conv_reID(tmp)
        tid = self.softmax(tid)
        tid = tid.squeeze(2)
        tid = torch.permute(tid, (0, 2, 1))
        
        return [torch.cat([heatmap, reg, wh, tid_embed], dim=1), tid]

if __name__ == '__main__':
 
    from dataloader import Mydata
    from torch.utils.data import DataLoader
   
    device = torch.device('cuda')
    path = '/home/thomas_yang/ML/hTG_MOT_pytorch/txt_file/mot16_02.txt'    
    dataset = Mydata(path)    
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0)
    
    for step, train_data in enumerate(dataloader):
        train_data = [i.to(device) for i in train_data]
        image, gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_tid, gt_tid_mask, tid_1d_idx = train_data
        if step == 0:
            break
    
    # x = torch.ones((1,3,416,416)).cuda()
    # tid = torch.ones((1, 100)).cuda()
    model = MyResNet50().cuda()
    y = model(image, tid_1d_idx)

    dummy_input = torch.randn(1, 3, 416, 416, device="cuda")
    input_names = [ "actual_input_1" , "tid_input"]
    output_names = [ "output1" ]
    torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=input_names, output_names=output_names)
