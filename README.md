# Unet Architecture for Image Segmentation

This repository contains the Unet architecture built with pytorch library. Here, the Unet is used to perform image segmentation task (work in progress).

##### Programmer: [Ravin Kumar](http://mr-ravin.github.io)
##### Repository: https://github.com/mr-ravin/unet-image-segmentation/

#### Software/Library Requirements:
  - Programming Language: Python 3
  - Deep Learning Library: Pytorch
  - Image Augmentation Library: Albumentations 
  - Image Processing Library: OpenCV
  - Other Libraries: tqdm

#### Directory Architecture:

```python3
  |-- dataset
  |    |-train/            # This directory contains training set.
  |    |-validate/         # This directory contains validation set.
  |
  |-- model.py               # file contains Unet Architecture in Pytorch Library.
  |-- datascript.py          # file contains code to access images of dataset for training and validation.
  |-- train.py               # file contains code for training Unet Architecture.
  |-- saved_models/          # This directory contains saved .pth file for Unet Architecture.
  

```


```python
Copyright (c) 2021 Ravin Kumar
Website: https://mr-ravin.github.io

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the 
Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
