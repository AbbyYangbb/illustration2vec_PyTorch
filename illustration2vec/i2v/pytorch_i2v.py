# This script (pytorch_i2v.py) is based on chainer_i2v.py and caffe_i2v.py.
# It build class PytorchI2V which can build PyTorch model, load weights, and load data. 

from i2v.base_1 import Illustration2VecBase
import os
import imp
import numpy as np
import json
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch.autograd import Variable



class PytorchI2V(Illustration2VecBase):
    def __init__(self, *args, **kwargs):
        super(PytorchI2V, self).__init__(*args, **kwargs)
        # caffe_mean = np.array([ 164.76139251,  167.47864617,  181.13838569])
        # we use pytorch mean and std on imagenet
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([0.229, 0.224, 0.225])
        self.mean = mean
        self.std = std
        self.model_path = kwargs['model_path']
        self.param_path = kwargs['param_path']   
        
    def _forward(self, inputs, layername):
        MainModel = imp.load_source('MainModel', self.model_path)
        model = torch.load(self.param_path)    
        inputs = self._image_loader(inputs)
        inputs = Variable(inputs)
        return model(inputs) # pool6

    def _image_loader(self, image_name): 
        # image size should be 224x224(resized after PIL opening)
        loader = T.Compose([
                            # T.Resize(imsize),
                            T.ToTensor(),
                            T.Normalize(mean = self.mean,
                                        std = self.std)])
        for img in image_name: # batch 
            img = img[:, :, [2, 1, 0]] # RGB -> BGR
            image_name = loader(img).unsqueeze(0)  
        return image_name.to("cpu", torch.float) # test with cpu

    def _extract(self, inputs, layername):
        # only test with layername = 'prob'
        if layername == 'prob':
            h = self._forward(inputs, layername='conv6_4') # layername: unused
            y = F.sigmoid(h)
            return y.data
        elif layername == 'encode1neuron':
            h = self._forward(inputs, layername='encode1')
            y = sigmoid(h)
            return y.data
        else:
            y = self._forward(inputs, layername)
            return y.data      
    
def make_i2v_with_pytorch(model_path=None, 
                          param_path=None, 
                          tag_path=None, 
                          threshold_path=None):
    kwargs = {}
    if model_path is not None:
        kwargs['model_path'] = model_path

    if param_path is not None:
        kwargs['param_path'] = param_path
    
    if tag_path is not None:
        tags = json.loads(open(tag_path, 'r').read())
        assert(len(tags) == 1539)
        kwargs['tags'] = tags

    if threshold_path is not None:
        fscore_threshold = np.load(threshold_path)['threshold']
        kwargs['threshold'] = fscore_threshold
    return PytorchI2V(**kwargs)

# if __name__ == '__main__':
#     model_path = './illustration2vec/illust2vec_pytorch.py'
#     model_weights_path = './illustration2vec/illust2vec_tag_ver200.pth'
#     tag_path = "/home/yang1489/Desktop/GAnime/i2v/illustration2vec/tag_list.json"
    
#     illust2vec = make_i2v_with_pytorch(model_path=model_path, model_weights_path=model_weights_path, tag_path=tag_path)
    
#     img_addr = '/home/yang1489/Desktop/GAnime/i2v/illustration2vec/images/miku.jpg'
#     img = Image.open(img_addr).convert('RGB')
#     img = img.resize((224, 224), Image.ANTIALIAS)
#     illust2vec.estimate_plausible_tags([img], threshold=0.5)

