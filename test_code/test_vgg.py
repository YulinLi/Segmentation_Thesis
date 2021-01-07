import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image

img_path = "/home/mislab/LiYulin/Research/dataset/LFW/train/image/Abdullah_Gul_0007.jpg"

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

batch_size = 8
feature_extract = True

class Vgg16(nn.Module):
    def __init__(self, pretrained = True):
        super(Vgg16, self).__init__()
        self.vggnet = models.vgg16(pretrained)
        del(self.vggnet.classifier) # Remove fully connected layer to save memory.
        features = list(self.vggnet.features)
        self.layers = nn.ModuleList(features).eval() 
        
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.layers):
            x = model(x)
            if ii in [29]:
                results.append(x) #(64,256,256),(128,128,128),(256,64,64),(512,32,32),(512,16,16)
                break
        return results

def bilitlp(feature_map, coord, img_width):
    x,y = coord[0], coord[1]
    _,_,H,W = feature_map.shape
    ratio = img_width/H
    feature_coord_y = int(y/ratio)
    feature_coord_x = int(x/ratio)
    w_y = int(y%ratio)
    w_x = int(x%ratio)
    
    left_top = feature_map[:,:,feature_coord_y,feature_coord_x]
    left_bot = feature_map[:,:,feature_coord_y + 1,feature_coord_x]
    rght_top = feature_map[:,:,feature_coord_y,feature_coord_x+1]
    rght_bot = feature_map[:,:,feature_coord_y+1,feature_coord_x+1]

    left_itpl = left_top* (ratio-w_y)/ratio + left_bot*(w_y)/ratio
    rght_itpl = rght_top* (ratio-w_y)/ratio + rght_bot*(w_y)/ratio
    bi_itpl = left_itpl* (ratio-w_x)/ratio + rght_itpl*(w_x)/ratio

    return bi_itpl

if __name__ == "__main__":
    img = Image.open(img_path ).convert('RGB')
    tensor_img = transforms.ToTensor()(img).unsqueeze_(0).cuda()
    batch,_,_,_ = tensor_img.shape
    vgg_model = Vgg16()
    vgg_model = vgg_model.cuda()
    #print(vgg_model.layers)
    #(70,60)
    feature_map = vgg_model(tensor_img)[0]
    tensor1 = (0)*feature_map[:,:,5,3] + (1)*feature_map[:,:,4,3]
    tensor2 = (0)*feature_map[:,:,5,4] + (1)*feature_map[:,:,4,4]
    tensor3 = (1)*tensor1 + (0)*tensor2
    tensor4 = bilitlp(feature_map, (64,48), 128)
    #print(tensor3.shape)
    print(tensor3)
    print(tensor4)
    print(feature_map[:,:,3,4])
    