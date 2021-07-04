import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from tqdm import tqdm
from model import UnetArchitecture

TRAIN_IMAGEDIR_PATH = "dataset/train/images/" 
TRAIN_MASKDIR_PATH = "dataset/train/masks/"
VAL_IMAGEDIR_PATH = "dataset/validate/images/"
VAL_MASKDIR_PATH =  "dataset/validate/masks/"

IMAGE_HT = 160
IMAGE_WT = 240
LOAD_MODEL = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 1
PIN_MEMORY = True # used when working with GPU

BATCH_SIZE = 16
EPOCHS = 100
LR = 10.0**-4     # learning rate is set to 0.0001

def train(loader,model,optimizer,loss_fn,scaler):
  loop = tqdm(loader)
  for batch_idx, (input_data,targets) in enumerate(loop):
    input_data = input_data.to(device=DEVICE) # shift to GPU if enabled
    targets = targets.float().unsqueeze().to(device=DEVICE) #shift to GPU if enabled.
    
    #  precision for GPU operations to improve performance while maintaining accuracy.
    with torch.cuda.amp.autocast():
      predictions = model(input_data)
      loss = loss_fn(predictions,targets)
      
    # backprop
    optimizer.zero_grad()
    scaler.scale(loss).backward() # gradient scaling
    scaler.step(optimizer)
    scaler.update()
    
    #update tqdm values
    loop.set_postfix(loss=loss.item())
    
def run():  # max_pixel_value=255.0 set to get value between 0.0 and 1.0 for pixel values 
  train_transform = A.Compose(
                [   A.Resize(height=IMAGE_HT, width=IMAGE_WT),
                    A.Rotate(limit=30,p=1.0),
                    A.HorizontalFlip(p=0.4),
                    A.VerticalFlip(p=0.2),
                    A.Normlize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0],max_pixel_value=255.0,),
                    ToTensorV2(),
                ],
                )

  validation_transform = A.Compose(
                [   A.Resize(height=IMAGE_HT, width=IMAGE_WT),
                    A.Normlize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0],max_pixel_value=255.0,),
                    ToTensorV2(),
                ],
                )
                
  model = UnetArchitecture(input_channels=3,output_channels=1).to(DEVICE)
  loss_fn = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(),lr=LR)
  
