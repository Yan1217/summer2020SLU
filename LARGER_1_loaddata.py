#!/usr/bin/env python
# coding: utf-8

# In[1]:
import keras
keras.__version__
import os, shutil,sys
import random


# In[ ]:

"""
original_dataset_dir = '/Users/yanru/Downloads/fake_real_image'
"""
real_orig = '/home/yanru/summer2020/data/real'
fake_orig = '/home/yanru/summer2020/data/fake'

train_dir = '/home/yanru/summer2020/LARGER/train'
test_dir = '/home/yanru/summer2020/LARGER/test'
validation_dir = '/home/yanru/summer2020/LARGER/validation'
train_real = '/home/yanru/summer2020/LARGER/train/real'
train_fake = '/home/yanru/summer2020/LARGER/train/fake'
real_validation = '/home/yanru/summer2020/LARGER/validation/real'
fake_validation = '/home/yanru/summer2020/LARGER/validation/fake'
test_real = '/home/yanru/summer2020/LARGER/test/real'
test_fake = '/home/yanru/summer2020/LARGER/test/fake'


list_real = os.listdir(real_orig)
seed=42
random_list_real = random.sample(list_real,len(list_real))
print("all real images: ",len(random_list_real))
real_len = len(random_list_real)
train_real_len = round(real_len * 0.7)
percent_real = round(real_len*0.15)

#copy images to training real
fnames = [random_list_real[i] for i in range (0,train_real_len)]
for fname in fnames:
    src = os.path.join(real_orig, fname)
    dst = os.path.join(train_real, fname)
    shutil.copyfile(src, dst)

print("train_real len:",len(os.listdir(train_real)))


# copy images to validation real
fnames = [random_list_real[i] for i in range (train_real_len,train_real_len+percent_real)]
for fname in fnames:
    src = os.path.join(real_orig, fname)
    dst = os.path.join(real_validation, fname)
    shutil.copyfile(src, dst)
    
print("validataion real len:",len(os.listdir(real_validation)))


# copy images to test real
fnames = [random_list_real[i] for i in range (train_real_len+percent_real,real_len)]
for fname in fnames:
    src = os.path.join(real_orig, fname)
    dst = os.path.join(test_real, fname)
    shutil.copyfile(src, dst)
    
print("rest real len:",len(os.listdir(test_real)))
    
    
    
# create a list of fake images 
list_fake = os.listdir(fake_orig)
#make is random
seed=42
random_list_fake = random.sample(list_fake,len(list_fake))

fake_lenth = len(random_list_fake)

print("all fake len:",fake_lenth)
train_fake_len = round(fake_lenth*0.7)
percent_fake = round(fake_lenth*0.15)


fnames = [random_list_fake[i] for i in range (0,train_fake_len)]
for fname in fnames:
    src = os.path.join(fake_orig, fname)
    dst = os.path.join(train_fake, fname)
    shutil.copyfile(src, dst)
print("train fake len:",len(os.listdir(train_fake)))

fnames = [random_list_fake[i] for i in range (train_fake_len,train_fake_len+percent_fake)]
for fname in fnames:
    src = os.path.join(fake_orig, fname)
    dst = os.path.join(fake_validation, fname)
    shutil.copyfile(src, dst)

print("validation fake len:",len(os.listdir(fake_validation)))

fnames = [random_list_fake[i] for i in range (train_fake_len+percent_fake,fake_lenth)]
for fname in fnames:
    src = os.path.join(fake_orig, fname)
    dst = os.path.join(test_fake, fname)
    shutil.copyfile(src, dst)

print("test fake len:",len(os.listdir(test_fake)))



print('Done')
