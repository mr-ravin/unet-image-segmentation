# Unet Architecture for Image Segmentation

This repository contains the Unet architecture built with pytorch library. Here, the Unet architecture is used to perform the image segmentation.

##### Programmer: [Ravin Kumar](http://mr-ravin.github.io)

#### Software/Library Requirements:
  - Programming Language: Python 3
  - Deep Learning Library: Pytorch
  - Image Augmentation Library: Albumentations 
  - Image Processing Library: OpenCV
  - Other Libraries: tqdm

###  Demonstration of Image Segmentation using Unet:

![Unet Image Segmentation](https://github.com/mr-ravin/unet-image-segmentation/blob/main/inference.jpg)

#### Directory Architecture:

```python3
  |-- dataset/
  |      |-train/            # This directory contains training set.
  |      |   |-images/
  |      |   |-masks/
  |      |
  |      |-validate/         # This directory contains validation set.
  |          |-images/
  |          |-masks/
  |
  |-- model.py               # file contains Unet Architecture in Pytorch Library.
  |-- datascript.py          # file contains code to access images of dataset for training and validation.
  |-- train.py               # file contains code for training Unet Architecture.
  |-- utils.py               # file contains model save, load, accuracy etc. related code.
  |-- saved_models/          # This directory contains saved .pth file for Unet Architecture.
  |-- saved_pred_images/     # This directory contains predicted segmented images.
  
```
###### Repository: https://github.com/mr-ravin/unet-image-segmentation/

Note: In our dataset we had .gif files as mask, so we first converted them to .jpg and then used OpenCV because .gif files have difficulty in opencv.
Terminal tool used for this conversion is: ```sudo apt install imagemagick```. A sample example is running in the terminal ```convert *.gif 1.jpg``` and also correspondingly changing the names of associated input images.

### Steps for Training the Unet Model:

```python3
python3 train.py
```

### Steps for Performing only Inference:
- Inside the train.py file, do set the following values:
  - ```LOAD_MODEL = 1     # set to 1 if you want to load weights and bias values of .pth file, else set it to 0.```
  - ```LOAD_MODEL_ID = 1  # set the model id present in saved_models/id_Unet_model.pth format (i.e. saved_models/1_Unet_model.pth).```
  - ```TRAINING = 0       # set to only perform inference.```

During Training and Inference part, all the output images will be saved inside the directory saved_pred_images/ 

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
