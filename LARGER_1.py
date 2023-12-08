#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
keras.__version__
import os, shutil,sys
import random


# In[1]:


original_dataset_dir = '/Users/yanru/Downloads/fake_real_image'
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


# In[ ]:


from keras import layers
from keras import models


# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[ ]:


from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
                    train_dir,target_size = (150,150),batch_size = 20,class_mode = 'binary')
test_generator = test_datagen.flow_from_directory(
                    test_dir,target_size = (150,150),batch_size = 20,class_mode = 'binary')


# In[ ]:


validation_generator = test_datagen.flow_from_directory(
                validation_dir,target_size = (150,150),batch_size = 20,class_mode = 'binary')


# In[ ]:


history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)


# In[ ]:


import pandas as pd
    
hist_df = pd.DataFrame(history.history) 


# or save to csv: 
hist_csv_file = 'history_large.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


# In[ ]:


test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
print('test loss:', test_loss)

