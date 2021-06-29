import torch
import torch.nn as nn
import torchvision.transforms.functional as TFN

class TwiceConv(nn.Module):
  def__init__(self,input_channels,output_channels):
    super(TwiceConv, self).__init__()
    self.conv_pair = nn.Sequential(
      nn.Conv2d(input_channels,output_channels,3,1,1,bias=False),
      nn.BatchNorm2d(output_channels),
      nn.ReLU(inplace = True),
      nn.Conv2d(output_channels,output_channels,3,1,1,bias=False),
      nn.ReLU(inplace = True)
    )
  def forward(self,x):
    return self.conv_pair(x)

class UnetArchitecture(nn.Module):
  def __init__(self,input_channels=3,output_channels=1,feature_list=[64,128,256,512]):
    super(UnetArchitecture,self).__init__()

    self.encoder_list = nn.ModuleList()
    self.decoder_list = nn.ModuleList()
    self.pool = nn.MaxPool2d(2,2)
    # encoder part  OR Down Part
    len_feature_list = len(feature_list)
    for idx in range(len_feature_list):
      self.encoder_list.append( TwiceConv(input_channels,feature_list[idx]) )
      input_channels = feature

    # decoder part OR Up Part
    for idx in range(len_feature_list): # reading feature_list in reverse order
      self.decoder_list.append(
              nn.ConvTranspose2d(
               feature_list[len_feature_list-idx]*2,feature_list[len_feature_list-idx],kernel_size=2,stride=2
               )
          )
      self.decoder_list.append(
             TwiceConv(feature_list[len_feature_list-idx]*2,feature_list[len_feature_list-idx])
          )

    self.bridge = TwiceConv(feature_list[-1],feature_list[-1]*2)
    self.decoder_last_conv = nn.Conv2d(feature_list[0],output_channels,kernel_size=1)


  def forward(self,x):
    connection_list = []

    for elem in self.encoder_list:
      x = elem(x)
      connection_list.append(x)
      x = self.pool(x)
    x = self.bridge(x)
    connection_list = connection_list[::-1]
