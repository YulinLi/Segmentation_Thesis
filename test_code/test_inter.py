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
            if ii in [3,8,15,22,29]:
                results.append(x) #(64,256,256),(128,128,128),(256,64,64),(512,32,32),(512,16,16)
        return results

if __name__ == "__main__":
    img = Image.open(img_path ).convert('RGB')
    tensor_img = transforms.ToTensor()(img).unsqueeze_(0).cuda()
    vgg_model = Vgg16()
    vgg_model = vgg_model.cuda()
    #print(vgg_model.layers)
    print((vgg_model(tensor_img)[4])[:,:,5,5].shape)