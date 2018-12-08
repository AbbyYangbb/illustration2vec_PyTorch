# Illustration2Vec

```illustration2vec (i2v)``` is the PyTorch implementation of paper: [Illustration2Vec: A Semantic Vector Representation of Illustrations](https://github.com/rezoo/illustration2vec/raw/master/papers/illustration2vec-main.pdf) from Masaki Saito and Yusuke Matsui. This model can estimate a set of tags and extracting semantic feature vectors from given illustrations. Our  ```i2v_pytorch``` model stucture and parameters are converted and modified from pre-trained caffe model proposed in this [repository](https://github.com/rezoo/illustration2vec). 

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
Here, we use an image of Miku\[1\] to show you how to use `i2v_pytorch` model.

![Hatsune Miku](https://github.com/Mukosame/GAnime/blob/i2v_branch/illustration2vec/images/miku.jpg?raw=true)

\[1]Hatsune Miku (初音ミク), © Crypton Future Media, INC., http://piapro.net/en_for_creators.html. This image is licensed under the Creative Commons - Attribution-NonCommercial, 3.0 Unported (CC BY-NC).
### Tag Prediction

`i2v` estimates a number of semantic tags from given illustrations in the following manner.

```python
import i2v
from PIL import Image
# if __name__ == '__main__':
model_path = '/your/path/to/model/illust2vec_tag_ver200.py'
param_path = '/your/path/to/params/illust2vec_tag_ver200.pth'
tag_path = "./tag_list.json"

illust2vec = i2v.make_i2v_with_pytorch(model_path=model_path, param_path=param_path, tag_path=tag_path)

img_addr = './images/miku.jpg'
img = Image.open(img_addr).convert('RGB')
img = img.resize((224, 224), Image.ANTIALIAS)
illust2vec.estimate_plausible_tags([img], threshold=0.5)

```

`estimate_plausible_tags()` returns dictionaries that have a pair of tag and its confidence.

```
[{'general': [('1girl', 1.0),
   ('solo', 1.0),
   ('long hair', 1.0),
   ('very long hair', 1.0),
   ('aqua hair', 1.0),
   ('twintails', 1.0),
   ('thighhighs', 1.0),
   ('skirt', 0.9999998807907104),
   ('detached sleeves', 0.9999997615814209),
   ('necktie', 0.9999994039535522),
   ('aqua eyes', 0.9999957084655762),
   ('headset', 0.9989746809005737),
   ('zettai ryouiki', 0.9972081780433655),
   ('simple background', 0.9719809889793396),
   ('thigh boots', 0.8656775951385498),
   ('boots', 0.5441169738769531)],
  'character': [('hatsune miku', 1.0)],
  'copyright': [('vocaloid', 1.0)],
  'rating': <zip at 0x7f7b5cc5de88>}]
```

