import keras
keras.__version__

import os, shutil,sys
import random

original_dataset_dir = '/home/yanru/summer2020/real_and_fake_face_orig'
training_r = '/home/yanru/summer2020/real_and_fake_face_orig/training_real'
training_f = '/home/yanru/summer2020/real_and_fake_face_orig/training_fake'
test_dir  = '/home/yanru/summer2020/real_and_fake_all/test'

train_dir = '/home/yanru/summer2020/real_and_fake_all/train'
validation_dir = '/home/yanru/summer2020/real_and_fake_all/validation'
train_real = '/home/yanru/summer2020/real_and_fake_all/train/training_real'
train_fake = '/home/yanru/summer2020/real_and_fake_all/train/training_fake'
real_validation = '/home/yanru/summer2020/real_and_fake_all/validation/validation_real'
fake_validation = '/home/yanru/summer2020/real_and_fake_all/validation/validation_fake'
test_real = '/home/yanru/summer2020/real_and_fake_all/test/testing_real'
test_fake = '/home/yanru/summer2020/real_and_fake_all/test/testing_fake'


# make a list of real imges 
list_real = os.listdir(training_r)
seed=42
random_list_real = random.sample(list_real,len(list_real))
#copy images to training real
fnames = [random_list_real[i] for i in range (0,500)]
for fname in fnames:
    src = os.path.join(training_r, fname)
    dst = os.path.join(train_real, fname)
    shuticl.copyfile(src, dst)

print("training real:", len(os.listdir(train_real)))

# copy images to validation real
fnames = [random_list_real[i] for i in range (500,800)]
for fname in fnames:
    src = os.path.join(training_r, fname)
    dst = os.path.join(real_validation, fname)
    shutil.copyfile(src, dst)
    
print("validation real:", len(os.listdir(real_validation)))


# copy images to test real
fnames = [random_list_real[i] for i in range (800,1081)]
for fname in fnames:
    src = os.path.join(training_r, fname)
    dst = os.path.join(test_real, fname)
    shutil.copyfile(src, dst)
print("test real:", len(os.listdir(test_real)))    

# create a list of fake images 
list_fake = os.listdir(training_f)
#make is random
seed=42
random_list_fake = random.sample(list_fake,len(list_fake))


print(random_list_fake[0])

nfnames = [random_list_fake[i] for i in range (0,400)]
for fname in fnames:
    src = os.path.join(training_f, fname)
    dst = os.path.join(train_fake, fname)
    shutil.copyfile(src, dst)

print("training fake:", len(os.listdir(train_fake)))

fnames = [random_list_fake[i] for i in range (400,700)]
for fname in fnames:
    src = os.path.join(training_f, fname)
    dst = os.path.join(fake_validation, fname)
    shutil.copyfile(src, dst)
print("validation fake:", len(os.listdir(fake_validation)))


fnames = [random_list_fake[i] for i in range (700,960)]
for fname in fnames:
    src = os.path.join(training_f, fname)
    dst = os.path.join(test_fake, fname)
    shutil.copyfile(src, dst)
print("test fake:", len(os.listdir(test_fake)))





