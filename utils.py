import torch
import torchvision
from datasetscript import CarvanaDataset
from torch.utils.data import DataLoader
import cv2 as cv

def save_weights(model_state,count,path="saved_models/"):
  torch.save(model_state,path+str(count)+"_Unet_model.pth")

def load_weights(model,count,path="saved_models/"):
  model.load_state_dict(torch.load(path+str(count)+"_Unet_model.pth"))


def check_accuracy(loader,model,device="cpu"):
  num_correct = 0
  num_pixels = 0.0
  dice_score = 0 # it gives the better understanding about accuracy of the model for segmentation
  model.eval()
  
  with torch.no_grad():
    for X,Y in loader:
      X = X.to(device)
      Y = Y.to(device).unsqueeze(1)    # output image have one channel.
      pred = torch.sigmoid(model(X))
      pred = (pred > 0.5).float()
      num_correct += (preds == Y).sum()
      num_pixels += torch.numel(preds)
      dice_score +=(2*(preds*Y).sum()) / ( (preds+Y).sum() + 1e-8 )
      print(str(num_correct/num_pixels)+" with accuracy: "+ f"{num_correct/num_pixels*100:.2f}")
      print(f"Dice score: {dice_score/len(loader)}")
      model.train()
      
      
def get_loaders(train_dir,train_maskdir,val_dir,val_maskdir,batch_size,train_transform,val_transform,num_workers=1,pin_memory=True):
  train_dataset = CarvanaDataset(imagedir_path=train_dir,maskdir_path=train_maskdir,transform=train_transform)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
  val_dataset = CarvanaDataset(imagedir_path=val_dir,maskdir_path=val_maskdir,transform=val_transform)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
  return train_loader, val_loader
  
def save_pred(loader, model, folder="saved_pred_images/", device="cpu"):
  model.eval() # it is same as:   model.train(mode=False)
  for idx, (x, y) in enumerate(loader):
    x = x.to(device=device)
    with torch.no_grad():
      preds = torch.sigmoid(model(x))
      preds = (preds > 0.5).float()
    cv.imwrite(preds,folder+str(idx)+"_pred.png")
  model.train()
