import cv2 as cv
import os
from torch.utils.data import Dataset

class CarvanaDataset(Dataset):
  def __init__(self, imagedir_path,maskdir_path,transform=None):
    self.imagedir_path = imagedir_path
    self.maskdir_path = maskdir_path
    self.transform = transform
    self.image_list = os.listdir(imagedir_path)
    
  def __len__(self):
    return len(self.image_list)
    
  def __getitem__(self,idx):
    image_path = imagedir_path+"/"+image_list[idx]            # filename with .jpg extension
    mark_path = maskdir_path+"/"+image_list[idx][:-4]+".gif"  # corresponding mask images have filename with .gif extension
    tmp_image = cv.imread(image_path)
    tmp_mask = cv.cvtColor(cv.imread(mask_path),cv.COLOR_BGR2GRAY) # convert to single channel gray scale image
    tmp_mask[tmp_mask==255.0] = 1.0  # converted the mask to a binary image
    
    image=tmp_image # image
    mask=tmp_mask   # corresponding mask of image
    
    if self.transform is not None:
      augmentations = self.transform(image=tmp_image,mask=tmp_mask)  # used to do transformations like scaling, rotation, flip etc.
      image=augmentations["image"]
      mask=augmentations["mask"]
      
    return image, mask
