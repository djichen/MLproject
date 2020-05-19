
# !/usr/bin/python3
# coding: utf-8
# @author: Deng Junwei
# @date: 2019/3/15
# @institute: JI@SJTU,UMSI
# @version: 0.1 alpha

import torch.utils.data as data
import torch
from torchvision import transforms
import numpy as np
import math
import copy
from utils_data import *
from PIL import Image
from config import *

class img_seg_ldr(torch.utils.data.Dataset):
        def __init__(self, transform = data_transform, data_dir = ".", upd_content = True, label_transform = label_transform):

            self.image_list, self.mask_list, self.filename_list = content_generator(data_dir, new=upd_content)
            self.SIZE = len(self.image_list) # calculate the filenum as the size of our data
            self.rootdir = data_dir 
            self.transform = transform
            self.label_transform = label_transform

        def __getitem__(self, idx):

            img_filename = self.image_list[idx]
            lable_filename = self.mask_list[idx]
            filename = self.filename_list[idx]

            img = Image.open(img_filename)
            label = Image.open(lable_filename)
            label = self.label_transform(label)
            label = generateLabel(label)

            # transform
            # label = torch.from_numpy(label).type(torch.FloatTensor) # Float for MSE/SmoothL1Loss()
            
            img = self.transform(img)
            label = torch.from_numpy(label).type(torch.LongTensor)

            return img, label, filename

        def __len__(self):
            return self.SIZE

class img_seg_ldr_test(torch.utils.data.Dataset):
        def __init__(self, transform = data_transform, data_dir = ".", upd_content = True, label_transform = label_transform):

            self.image_list, self.mask_list, self.filename_list = content_generator_test(data_dir, new=upd_content)
            self.SIZE = len(self.image_list) # calculate the filenum as the size of our data
            self.rootdir = data_dir 
            self.transform = transform
            self.label_transform = label_transform

        def __getitem__(self, idx):

            img_filename = self.image_list[idx]
            lable_filename = self.mask_list[idx]
            filename = self.filename_list[idx]

            img = Image.open(img_filename)
            label = Image.open(lable_filename)
            label = self.label_transform(label)
            label = generateLabel(label)

            # transform
            # label = torch.from_numpy(label).type(torch.FloatTensor) # Float for MSE/SmoothL1Loss()
            
            img = self.transform(img)
            label = torch.from_numpy(label).type(torch.LongTensor)

            return img, label, filename

        def __len__(self):
            return self.SIZE