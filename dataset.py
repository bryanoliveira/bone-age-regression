#!/usr/bin/env python
# coding: utf-8

BATCH_SIZE = 32
WORKERS = 4
IMG_SIZE = 320
GENDER_SENSITIVE = True
DS_MEAN = 0.1826
DS_STD = 0.1647


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torchvision import transforms

from skimage.exposure import equalize_adapthist

import PIL
from PIL import ImageOps, ImageEnhance


class BoneAgeDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, male=None, apply_transform=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            male (boolean): If image is from a man
            apply_transform (boolean): Enable random transformations (augmentation)
        """
        self.df_images = pd.read_csv(csv_file, usecols=['fileName', 'male'])
        self.df_labels = pd.read_csv(csv_file, usecols=['boneage', 'male'])
        
        self.root_dir = root_dir
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(30, translate=(0.1, 0.1), scale=(0.7, 1), shear=None, resample=False, fillcolor=0),
        ]) if apply_transform else None
        
        if male is not None:
            self.df_images = self.df_images[self.df_images['male'] == male]
            self.df_labels = self.df_labels[self.df_labels['male'] == male]
            
        self.df_images = self.df_images.drop('male', 1)
        self.df_labels = self.df_labels.drop('male', 1)

    def __len__(self):
        return len(self.df_images)

    def __getitem__(self, idx):        
        # read img file
        img_path = os.path.join(self.root_dir, self.df_images.iloc[idx, 0])
        img = load_image(img_path, self.transform)
        
        # read label
        label = self.df_labels.iloc[idx, 0]
        label = np.array([label])
        
        sample = {'images': img, 'labels': label}

        return sample


def load_image(img_path, transform=None):
    img = PIL.Image.open(img_path)
    img = transforms.functional.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=2)

    # CLAHE contrast enhancement
    img = np.array(img)
    img = equalize_adapthist(img)
    img = PIL.Image.fromarray(img)
    
    if transform is not None:
        img = transform(img)

    img = transforms.functional.to_tensor(img)
    return img

def generate_dataset(male):
    # prepare full dataset
    full_dataset = BoneAgeDataset('train.csv', 'images', male=male)

    # split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    
    return full_dataset, train_dataset, val_dataset, train_loader, val_loader


if __name__ == '__main__':
    # test dataset
    full_mixed_dataset, mixed_train_dataset, mixed_val_dataset, mixed_train_loader, mixed_val_loader = generate_dataset(None)
    print('Dataset length: ', len(full_mixed_dataset))
    print('Full ds item: ', full_mixed_dataset[0]['images'].shape, full_mixed_dataset[0]['labels'].shape)
    
    # test load_image:
    img = load_image('images/2904.png')
    print(img)
