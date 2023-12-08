#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras import layers
from keras import models
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
# In[10]:


train_dir = '/home/yanru/summer2020/SYETry/K_Cross_train'
train_data = pd.read_csv('C_K.csv',error_bad_lines=False)
print(train_data['label'])
l_l = train_data['label'].to_list()
#print(l_l)		
print(np.shape(train_data))
# Y is a panda's dataframe the shape is number of rows and 1 eg.(7958,1)
# it is the same thing for X 
Y = train_data[['label']]
X = train_data[['f_name']]
train_data['label'] = train_data['label'].astype(str)
#print("X",X)
#print(type(X),np.shape(X))
#print("Y",Y)
#print(type(Y),np.shape(Y))

kf = KFold(n_splits=5)
print(type(kf))

# split indcies into 5 sets for training and validation
for train_index, val_index in kf.split(X,Y):
	print("T", train_index)
	print("V", val_index)


skf = StratifiedKFold(n_splits = 5, random_state = 7, shuffle = True)
# In[7]:


datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
def build_model():	

       
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
    return model
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model
    """
# In[11]:

# a list of validation acc/loss for k number of runs 
validation_acc=[]
validation_loss=[]
fold_var = 1

# for each fold_var
for train_index, val_index in kf.split(X,Y):
    print(train_index)
    print(val_index)
    # selecting subset from training data
    # type panda data frame
    training_data = train_data.iloc[train_index]
    validation_data = train_data.iloc[val_index]
    print("T",training_data)
    print("Shape",np.shape(training_data)) 
   # fname_list = training_data['f_name'].tolist()
    #print(fname_list)

    print("V",validation_data)
    print("shapeV",np.shape(validation_data))
    train_generator = datagen.flow_from_dataframe(training_data, directory=train_dir,
                                                   x_col="f_name", y_col="label",batch_size=32,
                                                   class_mode="binary", shuffle=True,target_size=(150,150))
   
    validation_generator = datagen.flow_from_dataframe(validation_data, directory=train_dir,
                                                   x_col="f_name", y_col="label",batch_size = 32,
                                                   class_mode='binary', shuffle=True,target_size=(150,150))
    
    
    model = build_model()
   # model.compile(loss='categorical_crossentropy',
    #              optimizer=optimizers.RMSprop(lr=1e-4),
     #             metrics=['acc'])
    model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="binary_crossentropy",metrics=["accuracy"])
    

    history=model.fit_generator(
        train_generator,
        steps_per_epoch=2,
        epochs=50,
        validation_data=validation_generator,
        )
   # results = model.evaluate(validation_generator)
   # results = dict(zip(model,results))
    v_loss,v_acc=model.evaluate(validation_generator,steps=50)
   # validation_acc.append(results['accuracy'])
    validation_acc.append(v_acc)
    validation_loss.append(v_loss)
  # validation_loss.append(results['loss'])
    tf.keras.backend.clear_session()
    fold_var +=1
print("validation_acc", validation_acc)
print("validation_loss", validation_loss)
        

# In[ ]:




