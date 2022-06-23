from albumentations.augmentations.transforms import Transpose
from torch.utils.data import Dataset
from modules.utils import load_json
import numpy as np
from PIL import Image
import os
import pandas as pd
import torchvision
from torchvision import transforms
import cv2
import albumentations
import albumentations.pytorch
import albumentations.augmentations

class CowDataset(Dataset):
    def __init__(self, img_folder, dfpath):
        self.df = pd.read_csv(dfpath, usecols=['imname','grade'],dtype={'grade':str})
        self.label_encoding = {'1++':0, '1+':1, '1':2, '2':3, '3':4}
        self.img_folder = img_folder
        self.transform = albumentations.Compose([ 
            albumentations.RandomCrop(160, 160),
            albumentations.OneOf([
                              albumentations.HorizontalFlip(p=1),
                              albumentations.RandomRotate90(p=1),
                              albumentations.VerticalFlip(p=1)            
            ], p=1),
            albumentations.OneOf([
                              albumentations.MotionBlur(p=1),
                              albumentations.OpticalDistortion(p=1),
                              albumentations.GaussNoise(p=1)                 
            ], p=1),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
        
        self.image_names = self.df['imname']
        self.labels = self.df['grade']
        
    def __len__(self):
        return len(self.image_names)
        

    def __getitem__(self, index):
        impath = os.path.join(self.img_folder, self.image_names[index])
        image = cv2.imread(impath)       
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        image = augmented['image']
        target = self.labels[index]
        lbl = self.label_encoding[target]
        
        return image,lbl

class CowDataset_val(Dataset):
    def __init__(self, img_folder, dfpath):
        self.df = pd.read_csv(dfpath, usecols=['imname','grade'],dtype={'grade':str})
        self.label_encoding = {'1++':0, '1+':1, '1':2, '2':3, '3':4}
        self.img_folder = img_folder
        self.transform = albumentations.Compose([ 
        albumentations.RandomCrop(160, 160),
        albumentations.pytorch.transforms.ToTensorV2()
    ])
        
        self.image_names = self.df['imname']
        self.labels = self.df['grade']
        
    def __len__(self):
        return len(self.image_names)
        

    def __getitem__(self, index):
        impath = os.path.join(self.img_folder, self.image_names[index])
        image = cv2.imread(impath)       
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        image = augmented['image']
        target = self.labels[index]
        lbl = self.label_encoding[target]
        
        return image,lbl


class TestDataset(Dataset):
    def __init__(self, img_folder, dfpath):
        self.df = pd.read_csv(dfpath, usecols=['imname'],dtype={'imname':str})
        self.img_folder = img_folder
        self.transform = albumentations.Compose([ 
            albumentations.RandomCrop(160, 160),
            albumentations.OneOf([
                              albumentations.HorizontalFlip(p=1),
                              albumentations.RandomRotate90(p=1),
                              albumentations.VerticalFlip(p=1)            
            ], p=1),
            albumentations.OneOf([
                              albumentations.MotionBlur(p=1),
                              albumentations.OpticalDistortion(p=1),
                              albumentations.GaussNoise(p=1)                 
            ], p=1),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
        self.image_names = self.df['imname']
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        impath = os.path.join(self.img_folder, self.image_names[index])
        image = cv2.imread(impath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        image_test = augmented['image']
        filename = self.image_names[index]
        
        return image_test,filename
        
        
    
if __name__ == '__main__':
    pass

        