'''
Based on vgg11 model, we made two major modifications in the illust2vec_pytorch model:
(i) replace FC layers with convolutional layers; 
(ii) change softmax to sigmoid + cross-entropy. 
The change in (ii) can be found in '_extract' method of class PytorchI2V in pytorch_i2v.py; 
here the network's forward method returns result after pool6
'''

import torch as t
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np

# __weights_dict = dict()

# def load_weights(weight_file):
#     if weight_file == None:
#         return

#     try:
#         weights_dict = np.load(weight_file).item()
#     except:
#         weights_dict = np.load(weight_file, encoding='bytes').item()

#     return weights_dict

class I2V(t.nn.Module):
    def __init__(self):
        super(I2V, self).__init__()
#         global __weights_dict
#         __weights_dict = load_weights(weight_file)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),         
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),           
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),            
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),      
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1)                ,
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1539, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.AvgPool2d(kernel_size=(7,7), stride=(1,1)),
        )

    def forward(self, X):
        net = self.features(X)
        net = self.classifier(net)        
        return net # pool6
