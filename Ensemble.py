import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D,Flatten,Conv2D,BatchNormalization,Activation, MaxPooling2D
from keras.optimizers import Adam, SGD, RMSprop
p_size=48
folder_path="/content/"
expression='character_1_ka'
plt.figure(figsize=(12,12))
for i in range(1,10,1):
  plt.subplot(3,3,i)
  img=keras.preprocessing.image.load_img(folder_path+"train/"+expression+"/"+os.listdir(folder_path+"train/"+expression)[i+10],target_size=(p_size,p_size))
  plt.imshow(img)
plt.show()