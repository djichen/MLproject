
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
        def __init__(self, transform = data_transform, data_dir = ".", upd_content = True, Test = False):

            self.image_list, self.mask_list = content_generator(data_dir, new=upd_content, Test = Test)
            self.SIZE = len(self.image_list) # calculate the filenum as the size of our data
            self.rootdir = data_dir 
            self.transform = transform
            self.Test = Test

        def __getitem__(self, idx):

            img_filename = self.image_list[idx]
        
            img = Image.open(img_filename)
            if self.Test == False:
                lable_filename = self.mask_list[idx]
                label = Image.open(lable_filename)
                label = generateLabel(label)
                label = torch.from_numpy(label).type(torch.LongTensor)
                img = self.transform(img)
                return img, label

            img = self.transform(img)
            return img


        def __len__(self):
            return self.SIZE