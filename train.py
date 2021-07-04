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
    input_data = input_data.to(device=DEVICE)               # shift to GPU if enabled
    targets = targets.float().unsqueeze().to(device=DEVICE) # shift to GPU if enabled.
    
    #  precision for GPU operations to improve performance while maintaining accuracy.
    with torch.cuda.amp.autocast():
      predictions = model(input_data)
      loss = loss_fn(predictions,targets)
      
    # backprop
    optimizer.zero_grad()
    scaler.scale(loss).backward() # gradient scaling
    scaler.step(optimizer)
    scaler.update()
    
    # update tqdm values
    loop.set_postfix(loss=loss.item())
