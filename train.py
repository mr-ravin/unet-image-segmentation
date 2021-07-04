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

