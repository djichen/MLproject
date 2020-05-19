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



# Data Loader
dataset_test = img_seg_ldr_test()
test_loader = DataLoader(dataset_test, batch_size=1)

# Device identification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Try to find out if the computer have a CUDA with Nivida GPU, else we will use CPU to work

# Loss Function 
criterion = nn.CrossEntropyLoss(weight =torch.from_numpy(np.array([1,2,2,1])).type(torch.FloatTensor).to(device), reduction='sum')

# net = Nested_UNet(in_ch=3, out_ch=4)
net = Nested_UNet(in_ch=3, out_ch=4).to(device)

net.load_state_dict(torch.load('models/train_size=2000-epoch=11/model.pth'))
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