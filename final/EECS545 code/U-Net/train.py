# !/usr/bin/python3
# coding: utf-8
# @author: Deng Junwei
# @date: 2020/3/2
# @institute: UMSI
# @version: 0.1 alpha

# from unet_model import UNet
# from two_layer_unet_model import UNet
from three_layer_unet_model import UNet
import logging
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
num_epochs = 20
learning_rate = 0.01

# Data Loader
dataset_train = img_seg_ldr()
train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True)

# Device identification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Try to find out if the computer have a CUDA with Nivida GPU, else we will use CPU to work

# Model
net = UNet(n_channels=3, n_classes=4).to(device)

# Loss Function 
criterion = nn.CrossEntropyLoss(weight =torch.from_numpy(np.array([1,2,2,1])).type(torch.FloatTensor).to(device), reduction='sum')

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Save model:
dir_checkpoint = 'checkpoints/'

# Read from a current trained model:
start_epoch_num = 10

try:
    model_path = "checkpoints/three_layer_CP_epoch{}.pth".format(str(start_epoch_num-1))
    net.load_state_dict(torch.load(model_path))
except:
    print("We don't have the current trained model for epoch {}".format(start_epoch_num - 1))

# Training
net.train()
total_step = len(train_loader)
for epoch in range(start_epoch_num, num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 1 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                .format(epoch, num_epochs, i+1, total_step, loss.item()))
    try:
        os.mkdir(dir_checkpoint)
        logging.info('Created checkpoint directory')
    except OSError:
        pass
    torch.save(net.state_dict(),dir_checkpoint + f'three_layer_CP_epoch{epoch}.pth')
    logging.info(f'Checkpoint {epoch} saved !')

net.eval()
for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)
    outputs = net(images)
    loss = criterion(outputs, labels)
    print(loss.item())
    savepic(outputs[0,:], "{}_predict_three_layer.png".format(i))