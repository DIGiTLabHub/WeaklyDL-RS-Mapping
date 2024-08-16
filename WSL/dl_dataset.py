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
    else:
        temp = im_data
        maxx = np.max(temp)
        minn = np.min(temp)
    if maxx == minn:
        flag = 0
    else:
        flag = 1
    return flag

f0 = open('test.txt','r')
f1 = open('test_new.txt.txt','w')
for path in f0:
    path = path.rstrip()
    words = path.split()
    if 'DS_Store' in words[0]:
        continue
    flag = read_img(words[0])
    if flag == 1:
        f1.write(path+'\n')
