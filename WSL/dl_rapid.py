# import GeoTIFF package
from osgeo import gdal
from PIL import Image
import matplotlib.image as mpimg
from osgeo import ogr
import subprocess
from tensorflow import keras
import matplotlib.pyplot as plt
import IPython.display as display
import numpy as np
import pathlib
import tensorflow as tf
import torch
import os
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import random

transform = transforms.ToTensor()
#read the GeoTIFF file
def read_img(dataset_path):
    #print(dataset_path)
    dataset = gdal.Open(dataset_path)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)
    del dataset
    if im_bands == 13 :
        temp = np.delete(im_data,10,axis = 0)
        maxx = np.max(temp)
        minn = np.min(temp)
        temp = (temp - minn)/(maxx-minn)
    else:
        temp = im_data
        maxx = np.max(temp)
        minn = np.min(temp)
        temp = (temp - minn)/(maxx-minn)
    temp = np.transpose(temp,(1,2,0))
    final = np.resize(temp,(224,224,12))
    return final

class MyDataset(torch.utils.data.Dataset): #Create my dataset which inherits from torch.utils.data.Dataset
    def __init__(self,txt, transform=None, target_transform=None):
        super(MyDataset,self).__init__()
        path=txt
        file_list=open(path,'r')
        imgs = []
        for line in file_list:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, label = self.imgs[index]
        image = read_img(img)
        if self.transform is not None:
            image = self.transform(image) #transform images as we defined before
        return image,label

    def __len__(self):
        return len(self.imgs)

class  VGG(nn.Module):
    def __init__(self,num_classes=11):
        super(VGG,self).__init__()
        layers=[]
        in_dim=12
        out_dim=64
        #循环构造卷积层，一共有13个卷积层
        for i in range(13):
            layers+=[nn.Conv2d(in_dim,out_dim,3,1,1),nn.ReLU(inplace=True)]
            in_dim=out_dim
            #在第2、4、7、10、13个卷积后增加池化层
            if i==1 or i==3 or i==6 or i==9 or i==12:
                layers+=[nn.MaxPool2d(2,2)]
                #第10个卷积后保持和前边的通道数一致，都为512，其余加倍
                if i!=9:
                    out_dim*=2
        self.features=nn.Sequential(*layers)
        #VGGNet的3个全连接层，中间有ReLU与Dropout层
        self.classifier=nn.Sequential(

            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,num_classes),
        )
    def forward(self,x):
        x=self.features(x)
        #这里是将特征图的维度从[1,512,7,7]变到[1,512*7*7]
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x

train_data=MyDataset(txt='train_new.txt',transform=transform)
test_data=MyDataset(txt='test_new.txt',transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=16)

net = VGG()
net = net.double()
criterion =torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.05 )

acc_test = []
acc_train = []
loss_train = []
loss_test = []

for epoch in range(10):
    net.train()
    for i, data in enumerate(train_loader,0):
        img, label = data
        optimizer.zero_grad()
        output = net(img)
        prediction = torch.max(F.softmax(output), 1)[1]
        pred = prediction.data.numpy().squeeze()
        x = label.data.numpy().squeeze()
        acc_now = pred-x
        temp = np.sum(acc_now == 0)
        acc_train.append(temp/len(label))
        loss_contrastive = criterion(output,label)
        loss_contrastive.backward()
        optimizer.step()
        loss_train.append(loss_contrastive.item())
        print("train times: {}\nEpoch number {}\n Current loss {}\n Current accuracy {}\n".format(i,epoch,loss_contrastive.item(),temp/len(label)))
    else:
        net.eval()
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for i, data in enumerate(test_loader,0):
               img, label= data
               out = net(img)
               prediction = torch.max(F.softmax(out), 1)[1]
               x = label.data.numpy().squeeze()
               pred = prediction.data.numpy().squeeze()
               acc_now = pred-x
               temp = np.sum(acc_now == 0)
               acc.append(temp/len(label))
               loss_contrastive = criterion(out,label)
               loss_test.append(loss_contrastive.item())
label_to_index
