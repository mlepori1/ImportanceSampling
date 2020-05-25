import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

TRAIN=2000
TEST=500

class ColoredShapesDataset(Dataset):    
    """Dataset class for colored shapes images"""

    def __init__(self, data_dir, red_squares=0, red_triangles=0, blue_squares=0, blue_triangles=0, class_zero=[], transform=None, train=True):
        """
        Args:
            data_directory: Directory with all the images.
            red_squares: Number of Red Square Images
            red_triangles: Number of Red Triangle Images
            blue_squares: Number of Blue Square Images
            blue_triangles: Number of Blue Triangle Images
            class_zero: list of strings indicating which 
                shapes are of class 0, all others are class 1
            transform: Optional transform to be applied
                on a sample.
            train: boolean denoting whether this is a train 
                or test set
        """
        np.random.seed(9)
        self.data_dir = data_dir
        self.red_squares = red_squares
        self.red_triangles = red_triangles
        self.blue_squares = blue_squares
        self.blue_triangles = blue_triangles
        self.transform = transform
        self.train = train
        self.class_zero = class_zero
        self.data = []

        if self.train and red_squares > TRAIN:
            print(f'Requested more Red Square Images than exist in directory, providing {TRAIN} Red Square Images')
            self.red_squares=TRAIN
        if self.train and red_triangles > TRAIN:
            print(f'Requested more Red Triangle Images than exist in directory, providing {TRAIN} Red Triangle Images')
            self.red_triangles=TRAIN
        if self.train and blue_squares > TRAIN: 
            print(f'Requested more Blue Square Images than exist in directory, providing {TRAIN} Blue Square Images')
            self.blue_squares=TRAIN
        if self.train and blue_triangles > TRAIN:
            print(f'Requested more Blue Triangle Images than exist in directory, providing {TRAIN} Blue Triangle Images')
            self.blue_triangles=TRAIN
        if not self.train and red_squares > TEST:
            print(f'Requested more Red Square Images than exist in directory, providing {TEST} Red Square Images')
            self.red_squares=TEST
        if not self.train and red_triangles > TEST:
            print(f'Requested more Red Triangle Images than exist in directory, providing {TEST} Red Triangle Images')
            self.red_triangles=TEST
        if not self.train and blue_squares > TEST: 
            print(f'Requested more Blue Square Images than exist in directory, providing {TEST} Blue Square Images')
            self.blue_squares=TEST
        if not self.train and blue_triangles > TEST:
            print(f'Requested more Blue Triangle Images than exist in directory, providing {TEST} Blue Triangle Images')
            self.blue_triangles=TEST

        acceptable_classes = ['red_square', 'red_triangle', 'blue_square', 'blue_triangle']
        for cl in self.class_zero:
            if cl not in acceptable_classes:
                print("Unacceptable class found in class_zero argument")
                
        if self.blue_squares > 0:
            for i in range(self.blue_squares):
                if not self.train:
                    i +=  TRAIN
                img_name = os.path.join(self.data_dir,
                        f'Blue_Squares/image_{i}.png') 
                image = io.imread(img_name)
                if 'blue_square' in class_zero:
                    self.data.append({'image':image, 'class':0})
                else:
                    self.data.append({'image':image, 'class':1})

        if self.red_squares > 0:
            for i in range(self.red_squares):
                i += (TRAIN + TEST)
                if not self.train:
                    i += TRAIN
                img_name = os.path.join(self.data_dir,
                        f'Red_Squares/image_{i}.png') 
                image = io.imread(img_name)
                if 'red_square' in class_zero:
                    self.data.append({'image':image, 'class':0})
                else:
                    self.data.append({'image':image, 'class':1})

        if self.blue_triangles > 0:
            for i in range(self.blue_triangles):
                i += 2 * (TRAIN + TEST)
                if not self.train:
                    i += TRAIN
                img_name = os.path.join(self.data_dir,
                        f'Blue_Triangles/image_{i}.png') 
                image = io.imread(img_name)
                if 'blue_triangle' in class_zero:
                    self.data.append({'image':image, 'class':0})
                else:
                    self.data.append({'image':image, 'class':1})

        if self.red_triangles > 0:
            for i in range(self.red_triangles):
                i += 3 * (TRAIN + TEST)
                if not self.train:
                    i += TRAIN
                img_name = os.path.join(self.data_dir,
                        f'Red_Triangles/image_{i}.png') 
                image = io.imread(img_name)
                if 'red_triangle' in class_zero:
                    self.data.append({'image':image, 'class':0})
                else:
                    self.data.append({'image':image, 'class':1})
        
        
        self.data = np.array(self.data)
        np.random.shuffle(self.data)
        

    def __len__(self):
        return self.red_squares + self.red_triangles + self.blue_squares + self.blue_triangles

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, cl = sample['image'], sample['class']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        # Remove Alpha channel
        image = image[:3]
        return {'image': torch.from_numpy(image).float(),
                'class': torch.from_numpy(np.array(cl)).float()}