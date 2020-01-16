#!/usr/bin/env python
# coding: utf-8

# In[8]:


EPOCHS = 20
GRAD_NORM_CLIP = 0.1
LEARNING_RATE = 1e-3
GENDER_SENSITIVE = True


# In[16]:


import os
import collections
import datetime
import math
import numpy as np # linear algebra
import tqdm
import pandas as pd

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from model import Model, make_layers, cfg
from dataset import load_image, generate_dataset
from radam import RAdam


# # Load Dataset

# In[10]:


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


# # Train

# In[18]:


def generate_model():
    model = Model(make_layers(cfg['B']))
    model.cuda()

    optimizer = RAdam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    
    return model, optimizer, scheduler


# In[19]:


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


# In[13]:


def save_model(experiment_name, model, optimizer, scheduler, epoch, train_loss, val_loss):
    checkpoint = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    }
    torch.save(checkpoint, 'models/' + experiment_name + '_' + str(datetime.datetime.now()) + '.pt')
    print('Model ' + experiment_name + ' saved.')
    
def train(experiment_name, model, optimizer, scheduler, train_loader, val_loader):
    train_loss_hist = collections.deque(maxlen=500)
    val_loss_hist = collections.deque(maxlen=500)

    model.training = True
    model.train()
    model.freeze_bn()
    
    best_val_loss = 1e6
    
    writer = SummaryWriter()

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

            loss = float(loss)
            
            train_loss_hist.append(loss)

            epoch_loss.append(loss)
            
            progress.set_description(
                desc='Train - Ep: {} | It: {} | Ls: {:1.3f} | mLs: {:1.3f} | MAE: {:1.3f}'.format(
                    epoch_num, 
                    iter_num, 
                    loss, 
                    np.mean(train_loss_hist),
                    math.sqrt(loss)
                )
            )
            
            progress.update(1)

            del loss

        train_loss = np.mean(train_loss_hist)
        train_mae = math.sqrt(train_loss)
        
        writer.add_scalar('loss/train_loss_mean', train_loss, epoch_num)
        writer.add_scalar('loss/train_mae', train_mae, epoch_num)
        
        print('Train - Ep: {} | Ls: {:1.3f} | MAE: {:1.3f}'.format(epoch_num, train_loss, train_mae))
        
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

        val_loss = np.mean(val_loss_hist)
        val_mae = math.sqrt(val_loss)
        
        writer.add_scalar('loss/val_loss', val_loss, epoch_num)
        writer.add_scalar('loss/val_mae', val_mae, epoch_num)
        
        print('Val - Ep: {} | Ls: {:1.5f} | MAE: {:1.3f}'.format(epoch_num, val_loss, val_mae))

        scheduler.step(np.mean(epoch_loss))
        
        if val_loss < best_val_loss:
            save_model('checkpoint_' + experiment_name, model, optimizer, scheduler, epoch_num, train_loss, val_loss)
            best_val_loss = val_loss

    model.training = False
    model.eval()
    save_model('_final_' + experiment_name, model, optimizer, scheduler, EPOCHS - 1, np.mean(train_loss_hist), np.mean(val_loss_hist))
    
    writer.close()
    
    return model, optimizer, scheduler


# In[14]:


if not GENDER_SENSITIVE:
    print('\nTRAINING MIXED MODEL')
    mixed_model, mixed_optimizer, mixed_scheduler = train('mixed', mixed_model, mixed_optimizer, mixed_scheduler, mixed_train_loader, mixed_val_loader)
else:
    #print('\nTRAINING MALE MODEL')
    #male_model, male_optimizer, male_scheduler = train('male', male_model, male_optimizer, male_scheduler, male_train_loader, male_val_loader)
    print('\nTRAINING FEMALE MODEL')
    female_model, female_optimizer, female_scheduler = train('female', female_model, female_optimizer, female_scheduler, female_train_loader, female_val_loader)


# # Final Validation & Loading

# In[16]:


if False: # if LOAD
    def load(path):
        checkpoint = torch.load(path)
        return checkpoint

    male_model = load('models/_final_male_2020-01-16 05:39:29.190551.pt')
    female_model = load('models/_final_female_2020-01-16 08:05:38.533981.pt')

    male_model.cuda()
    female_model.cuda()


# # Generate Output

# In[20]:


test_df = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')
progress = tqdm.tqdm(total=len(test_df), desc='Sample', position=0)
for key, row in test_df.iterrows():
    img_path = os.path.join('images', row['fileName'])
    img = load_image(img_path)
    img = img.unsqueeze(0)
    img = img.cuda()
    
    if not GENDER_SENSITIVE:
        boneage = mixed_model(img)
    elif row['male'] == True:
        boneage = male_model(img)
    else:
        boneage = female_model(img)
        
    boneage = float(boneage.view(-1).detach().cpu()[0])
    submission.loc[submission.fileName == row['fileName'], 'boneage'] = boneage if boneage > 0 else 0
    progress.update(1)
    
submission.head()


# In[19]:


submission.to_csv('submission.csv', index=False)
print('Saved submission file.')
