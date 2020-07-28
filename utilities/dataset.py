import pandas as pd
import torch
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms

from utilities import data


class CIFAR10():
    def __init__(self, dataset, master, remove_unlabel=False):
        self.dataset = dataset
        self.df = pd.read_csv(master)
        self.remove_unlabel = remove_unlabel
        if self.remove_unlabel:
            unlabel_idx = self.df.index[self.df['unlabeled'] == True].tolist()
            self.df.drop(unlabel_idx , inplace=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = Image.open(self.df.iloc[idx,1])
        target = self.df.iloc[idx, 4]
        transform = data.transform(self.dataset)

        if self.dataset == 'train':
            image1 = transform(image)
            image2 = transform(image)
            return image1, image2, target
    
        else:
            image = transform(image)
            return image, target

    def get_index(self):
        unlabel_idx = self.df.index[self.df['unlabeled'] == True].tolist()
        label_idx = self.df.index[self.df['unlabeled'] == False].tolist()
        return unlabel_idx, label_idx
