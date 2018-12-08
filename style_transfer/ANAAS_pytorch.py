import numpy as np
import torch as t
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from skimage import io
import os

class my_Vgg19(t.nn.Module):
    def __init__(self):
        super(my_Vgg19, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    
    def forward(self, X):
        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.relu(self.conv3_4(h))
        relu3_4 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.relu(self.conv4_4(h))
        relu4_4 = h
        
        return [relu1_2, relu2_2, relu3_4, relu4_4]

def init_vgg19(model_folder):
    if not os.path.exists(os.path.join(model_folder, 'vgg19.weight')):
        my_vgg = my_Vgg19()
        vgg19 = models.vgg19(pretrained=True)
        for (src, dst) in zip(vgg19.parameters(), my_vgg.parameters()):
            dst.data[:] = src
        t.save(my_vgg.state_dict(), os.path.join(model_folder, 'vgg19.weight'))

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def DeNormalize(obj):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for c, m, s in zip(obj, mean, std):
        c.mul_(s).add_(m)
    return obj

content_image = io.imread('./content.jpg')
style_image = io.imread('./style.jpg')

content_image = content_image.cuda()
style_image = style_image.cuda()

transform = T.Compose([T.ToTensor(), 
                       T.Normalize(mean = [0.485, 0.456, 0.406], 
                                   std = [0.229, 0.224, 0.225])])
content_image = transform(content_image).float()
style_image = transform(style_image).float()
content_image = content_image.unsqueeze(0)
style_image = style_image.unsqueeze(0)

vgg = my_Vgg19()
init_vgg19('./models')
vgg.load_state_dict(t.load('./models/vgg19.weight'))
vgg.cuda()


features_content = vgg(content_image)
f_xc_c = Variable(features_content[-1].data, requires_grad=False)
features_style = vgg(style_image)
gram_style = [gram_matrix(y) for y in features_style]


output = Variable(content_image.data, requires_grad=True)
optimizer = Adam([output], lr=0.001)
mse_loss = nn.MSELoss()
content_weight = 1.0
style_weight = 5.0


for i in range(500):
    print(i)
    optimizer.zero_grad()
    features_y = vgg(output)
    content_loss = content_weight * mse_loss(features_y[-1], f_xc_c)
    
    style_loss = 0
    for m in range(len(features_y)):
        gram_y = gram_matrix(features_y[m])
        gram_s = Variable(gram_style[m].data, requires_grad=False)
        style_loss += style_weight * mse_loss(gram_y, gram_s)
    
    total_loss = content_loss + style_loss
    total_loss.backward()
    optimizer.step()


output = DeNormalize(output).squeeze(0)


# import matplotlib.pyplot as plt
# plt.imshow(output.data.numpy().transpose((1, 2, 0)))
# print(output.shape)

