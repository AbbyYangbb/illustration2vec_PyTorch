# Illustration2Vec

This is a PyTorch implementation of ```illustration2vec (i2v)``` model in paper: Illustration2Vec: A Semantic Vector Representation of Illustrations from Masaki Saito and Yusuke Matsui. This model can estimate a set of tags and extracting semantic feature vectors from given illustrations. Our  ```i2v_pytorch``` model stucture and parameters are converted and modified from pre-trained caffe model proposed in this [repository](https://github.com/rezoo/illustration2vec). 

## Requirements

Libraries: 

* ```numpy``` and ```scipy```
* `PIL` (Python Imaging Library) or its alternatives (e.g., `Pillow`)
* `skimage` (Image processing library for python)
* `PyTorch` (0.4.0) 

Pre-treined models:

Our `i2v_pytorch` model uses Convolutional Neural Networks - VGG 11 with following changes in network structure:

It contains only convolutional layers and a global average pooling layer. 

As for feature extracter, it keeps 11 convolutional layers in original VGG model. For classifier, three convolutional layers take the place of three fully-connected layers, and a global average layer is added. Furthermore, the softmax layer was replaced by the sigmoid layer, and the network is trained by minimizing the cross-entropy loss function. 

## How To Use
Here, we use an image of Sophie\[1\] from one of my favourite movie ```Howl's Moving Castle``` to show you how to use `i2v_pytorch` model.

![Sophie](https://github.com/Mukosame/GAnime/blob/i2v_branch/illustration2vec/images/Sophie_howls_moving_castle.png?raw=true)

\[1]Howl's Moving Castle ([wiki](https://en.wikipedia.org/wiki/Howl%27s_Moving_Castle_(film)))(Japanese: ハウルの動く城 Hepburn: Hauru no Ugoku Shiro) is a 2004 Japanese animated fantasy film written and directed by Hayao Miyazaki. 
### Tag Prediction

`i2v` estimates a number of semantic tags from given illustrations in the following manner.

```python
import i2v
from PIL import Image
# if __name__ == '__main__':
model_path = '/your/path/to/model/illust2vec_pytorch.py'
param_path = '/your/path/to/params/illust2vec_tag_ver200.pth'
tag_path = "./tag_list.json"

illust2vec = i2v.make_i2v_with_pytorch(model_path=model_path, param_path=param_path, tag_path=tag_path)

img_addr = './images/Sophie_howls_moving_castle.png'
img = Image.open(img_addr).convert('RGB')
img = img.resize((224, 224), Image.ANTIALIAS)
illust2vec.estimate_plausible_tags([img], threshold=0.5)

```

`estimate_plausible_tags()` returns dictionaries that have a pair of tag and its confidence.

```
[{'general': [('1girl', 1.0),
   ('solo', 1.0),
   ('face', 0.9999932050704956),
   ('smile', 0.9999369382858276),
   ('short hair', 0.9892030358314514),
   ('brown eyes', 0.9779762625694275),
   ('white hair', 0.6438878178596497),
   ('looking at viewer', 0.5579541921615601)],
  'character': [],
  'copyright': [('strike witches', 0.999961256980896)],
  'rating': <zip at 0x7fac21008b48>}]
```

We also have a video demo about this model. [Watch us on Youtube!](https://www.youtube.com/watch?v=ifVNqG2IwbI&feature=youtu.be)
(Thanks [Xiaoyu](https://github.com/Mukosame) for coming up this idea and editing the video!)
