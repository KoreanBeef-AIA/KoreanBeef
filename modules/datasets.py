from torch.utils.data import Dataset
from modules.utils import load_json
import numpy as np
from PIL import Image
import os
import pandas as pd
import torchvision.transforms as transforms

class CowDataset(Dataset):
    def __init__(self, img_folder, dfpath):
        self.df = pd.read_csv(dfpath, usecols=['imname','grade'],dtype={'grade':str})
        self.label_encoding = {'1++':0, '1+':1, '1':2, '2':3, '3':4}
        self.img_folder = img_folder
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])
        
        self.image_names = self.df['imname']
        self.labels = self.df['grade']
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        impath = os.path.join(self.img_folder, self.image_names[index])
        img = Image.open(impath)
        img = self.transforms(img)
        target = self.labels[index]
        lbl = self.label_encoding[target]
        
        return img,lbl

class TestDataset(Dataset):
    def __init__(self, img_folder, dfpath):
        self.df = pd.read_csv(dfpath, usecols=['imname'],dtype={'imname':str})
        self.img_folder = img_folder
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])
        self.image_names = self.df['imname']
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        impath = os.path.join(self.img_folder, self.image_names[index])
        img = Image.open(impath)
        img = self.transforms(img)
        filename = self.image_names[index]
        
        return img,filename
        
        
    
if __name__ == '__main__':
    pass

        