#!/usr/bin/env python
# coding: utf-8

# In[1]:


EPOCHS = 20
GRAD_NORM_CLIP = 0.1
BATCH_SIZE = 32
WORKERS = 4
LEARNING_RATE = 1e-3
IMG_SIZE = 256
GENDER_SENSITIVE = False


# In[3]:


# %matplotlib inline

import os
import collections
import datetime
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from skimage import io, transform
from sklearn.model_selection import KFold

from matplotlib import pyplot as plt


# # Dataset Analysis

# In[4]:


"""
df = pd.read_csv('train.csv')
df = df[df['male'] == True]
df.head()
"""


# In[5]:


# len(df)


# In[6]:


"""
img = io.imread('images/15133.png')
img = transform.resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
plt.imshow(img)
plt.show()
"""


# # Dataset Generation

# In[7]:


class BoneAgeDataset(Dataset):
    def __init__(self, csv_file, root_dir, male=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            male (boolean): If image is from a man
        """
        self.df_images = pd.read_csv(csv_file, usecols=['fileName', 'male'])
        self.df_labels = pd.read_csv(csv_file, usecols=['boneage', 'male'])

        self.root_dir = root_dir
        
        if male is not None:
            self.df_images = self.df_images[self.df_images['male'] == male]
            self.df_labels = self.df_labels[self.df_labels['male'] == male]
            
        self.df_images = self.df_images.drop('male', 1)
        self.df_labels = self.df_labels.drop('male', 1)

    def __len__(self):
        return len(self.df_images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df_images.iloc[idx, 0])
        img = io.imread(img_path)
        img = transform.resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
        img = img.reshape(-1, IMG_SIZE, IMG_SIZE)
        
        label = self.df_labels.iloc[idx, 0]
        label = np.array([label])
        #label = label.astype('float').reshape(-1, 2)
        
        sample = {'images': img, 'labels': label}

        return sample


# In[8]:


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


# In[15]:


if not GENDER_SENSITIVE:
    # prepare full dataset
    full_mixed_dataset, mixed_train_dataset, mixed_val_dataset, mixed_train_loader, mixed_val_loader = generate_dataset(None)
    print('Dataset length: ', len(full_mixed_dataset))
    print('Full ds item: ', full_mixed_dataset[0]['images'].shape, full_mixed_dataset[0]['labels'].shape)

else:
    # prepare male dataset
    full_male_dataset, male_train_dataset, male_val_dataset, male_train_loader, male_val_loader = generate_dataset(True)
    print('Male dataset length: ', len(full_male_dataset))
    print('Male ds item: ', full_male_dataset[0]['images'].shape, full_male_dataset[0]['labels'].shape)

    # prepare female dataset
    full_female_dataset, female_train_dataset, female_val_dataset, female_train_loader, female_val_loader = generate_dataset(False)
    print('Female dataset length: ', len(full_female_dataset))
    print('Female ds item: ', full_female_dataset[0]['images'].shape, full_female_dataset[0]['labels'].shape)


# # Model Definition
# 
# Adapted from: 
# - https://github.com/yhenon/pytorch-retinanet
# - https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
# - https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
# - https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py

# In[16]:


class Model(nn.Module):

    def __init__(self, features):
        super(Model, self).__init__()
        
        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.SELU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.SELU(True),
            nn.Linear(512, 1),
        )
        
        self.loss = nn.MSELoss()

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.freeze_bn()
        self.training = False

    def forward(self, inputs):

        if self.training:
            x, y = inputs
        else:
            x = inputs

        y1 = self.features(x)
        y1 = self.avgpool(y1)
        y1 = y1.view(y1.size(0), -1)
        y1 = self.classifier(y1)
        
        if self.training:
            return self.loss(y1, y)
        else:
            return y1

   
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                
def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.SELU(inplace=True)]
            else:
                layers += [conv2d, nn.SELU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
    
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


# # Train

# In[17]:


def generate_model():
    model = Model(make_layers(cfg['B']))
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    
    return model, optimizer, scheduler


# In[18]:


if not GENDER_SENSITIVE:
    # full mixed (male/female) model
    mixed_model, mixed_optimizer, mixed_scheduler = generate_model()
    print(mixed_model)
    
else:
    # male model
    male_model, male_optimizer, male_scheduler = generate_model()
    
    # female model
    female_model, female_optimizer, female_scheduler = generate_model()
    
    print(male_model) # print only one since they're equal


# In[19]:


def save_model(experiment_name, model, optimizer, scheduler, epoch, train_loss, val_loss):
    checkpoint = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    }
    torch.save(model, 'models/' + experiment_name + '_' + str(datetime.datetime.now()) + '.pt')
    print('Model ' + experiment_name + ' saved.')
    
def train(experiment_name, model, optimizer, scheduler, train_loader, val_loader):
    train_loss_hist = collections.deque(maxlen=500)
    val_loss_hist = collections.deque(maxlen=500)

    model.training = True
    model.train()
    model.freeze_bn()

    for epoch_num in range(EPOCHS):
        epoch_loss = []

        
        progress = tqdm.tqdm(total=len(train_loader), desc='Training Status', position=0)
        for iter_num, data in enumerate(train_loader):
            optimizer.zero_grad()

            loss = model([data['images'].cuda().float(), data['labels'].cuda().float()])

            loss = loss.mean()

            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM_CLIP)

            optimizer.step()

            train_loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            progress.set_description(
                desc='Train - Ep: {} | It: {} | Ls: {:1.3f} | mLs: {:1.3f} | MAE: {:1.3f}'.format(
                    epoch_num, 
                    iter_num, 
                    float(loss), 
                    np.mean(train_loss_hist),
                    math.sqrt(float(loss))
                )
            )
            progress.update(1)

            del loss

        print('Train - Ep: {} | Ls: {:1.3f} | MAE: {:1.3f}'.format(epoch_num, np.mean(train_loss_hist), math.sqrt(np.mean(train_loss_hist))))

        progress = tqdm.tqdm(total=len(val_loader), desc='Validation Status', position=0)
        for iter_num, data in enumerate(val_loader):
            with torch.no_grad():
                val_loss = model([data['images'].cuda().float(), data['labels'].cuda().float()])
                val_loss.mean()
                val_loss_hist.append(float(val_loss))
                optimizer.zero_grad()

                progress.set_description(
                    desc='Val - Ep: {} | It: {} | Ls: {:1.5f} | mLs: {:1.5f} | MAE: {:1.3f}'.format(
                        epoch_num, 
                        iter_num, 
                        float(val_loss), 
                        np.mean(val_loss_hist),
                        math.sqrt(float(val_loss))
                    )
                )
                progress.update(1)

        print('Val - Ep: {} | Ls: {:1.5f} | MAE: {:1.3f}'.format(epoch_num, np.mean(val_loss_hist), math.sqrt(np.mean(val_loss_hist))))

        scheduler.step(np.mean(epoch_loss))
        save_model('checkpoint_' + experiment_name, model, optimizer, scheduler, epoch_num, np.mean(train_loss_hist), np.mean(val_loss_hist))

    model.training = False
    model.eval()
    save_model('_final_' + experiment_name, model, optimizer, scheduler, EPOCHS - 1, np.mean(train_loss_hist), np.mean(val_loss_hist))
    
    return model, optimizer, scheduler


# In[20]:


if not GENDER_SENSITIVE:
    print('\nTRAINING MIXED MODEL')
    mixed_model, mixed_optimizer, mixed_scheduler = train('mixed', mixed_model, mixed_optimizer, mixed_scheduler, mixed_train_loader, mixed_val_loader)
else:
    print('\nTRAINING MALE MODEL')
    male_model, male_optimizer, male_scheduler = train('male', male_model, male_optimizer, male_scheduler, male_train_loader, male_val_loader)
    print('\nTRAINING FEMALE MODEL')
    female_model, female_optimizer, female_scheduler = train('female', female_model, female_optimizer, female_scheduler, female_train_loader, female_val_loader)


# # Generate Output

# In[44]:


test_df = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')
progress = tqdm.tqdm(total=len(test_df), desc='Sample', position=0)
for key, row in test_df.iterrows():
    img_path = os.path.join('images', row['fileName'])
    img = io.imread(img_path)
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    img = img.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
    img = torch.Tensor(img).cuda()
    
    if not GENDER_SENSITIVE:
        boneage = mixed_model(img)
    elif row['male'] == True:
        boneage = male_model(img)
    else:
        boneage = female_model(img)
        
    boneage = float(boneage.view(-1).detach().cpu()[0])
    submission.loc[submission.fileName == row['fileName'], 'boneage'] = boneage
    progress.update(1)
    
submission.head()


# In[45]:


submission.to_csv('submission.csv', index=False)
print('Saved submission file.')

