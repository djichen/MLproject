# !/usr/bin/python3
# coding: utf-8
# @author: Deng Junwei
# @date: 2020/3/2
# @institute: UMSI
# @version: 0.1 alpha

from unetpp import *
# from unet_model import UNet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import time
from data_ldr import *
from utils_predict import *
# Argument
num_epochs = 10
learning_rate = 0.01
training_size_threshold = 100

# Data Loader
dataset_train = img_seg_ldr()
dataset_test = img_seg_ldr_test()
train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=1)

# Device identification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Try to find out if the computer have a CUDA with Nivida GPU, else we will use CPU to work

# Model
# net = UNet(n_channels=3, n_classes=4).to(device)
net = Nested_UNet(in_ch=3, out_ch=4).to(device)

# Loss Function 
criterion = nn.CrossEntropyLoss(weight =torch.from_numpy(np.array([1,2,2,1])).type(torch.FloatTensor).to(device), reduction='sum')

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Training
net.train()
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels,filename) in enumerate(train_loader):
        if i> training_size_threshold:
	        break
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 1 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    torch.save(net.state_dict(),"models/model_epoch={}".format(epoch))

net.eval()
for i, (images, labels, filename) in enumerate(test_loader):
    filename = filename[0]
    filename = filename[0:-4]    
    images = images.to(device)
    labels = labels.to(device)
    outputs = net(images)
    loss = criterion(outputs, labels)
    print(loss.item())
    print("The {} prediction.".format(filename))
    savepic(outputs[0,:], "{}_predict.png".format(filename))