

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from scipy.interpolate import interp1d
from tensorflow.keras.models import Model
from tensorflow.keras.layers import UpSampling1D,Activation,LeakyReLU,Dense,GlobalAveragePooling1D ,BatchNormalization,Dropout,Flatten
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam
from math import ceil
from tensorflow.keras.layers import Conv1D ,AveragePooling1D,LSTM,TimeDistributed,Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import MaxPooling1D
import tensorflow.keras
### Get Reproducible Results with Keras #####
from numpy.random import seed
seed(42)
import tensorflow
tensorflow.random.set_seed(2)
np.random.seed(2)
from numpy import save
from numpy import load

# def regression_BiLSTM(timestep):
#     dropR=0.2
#     INp = keras.Input(shape=(timestep,2))
#     x = Bidirectional(LSTM(64,return_sequences=True))(INp)
#     x = Bidirectional(LSTM(32,return_sequences=True))(x)
#     #x= TimeDistributed(dense(25))(x)
#     x = Flatten()(x)
#     x = Dense(16)(x)
#     x = Activation(activation='relu')(x)
#     x = Dense(8)(x)
#     x = Activation(activation='relu')(x)
#     x = Dense(2)(x)
#     x = Activation(activation='linear')(x)
#     output = Model(INp , x, name='regression_1Dcnn')
#     return output

# model=regression_BiLSTM(40)
# model.summary()

# def regression_BiLSTMProposed(timestep):
#     dropR=0.5
#     INp = keras.Input(shape=(timestep,2))
#     x = Bidirectional(LSTM(64))(INp)
#     x = Dropout(dropR)(x)
    
#     x = Dense(2)(x)
#     x = Activation(activation='linear')(x)
#     output = Model(INp , x, name='regression_1Dcnn')
#     return output

# model=regression_BiLSTMProposed(40)
# model.summary()


listLR1=[1450,2050,2800]

listLRas=[2400,2000,1800,1500,1300,1100,800]
print(type(listLRas))

print(listLR1[0])
rang20=np.linspace(listLR1[0],listLR1[2],10)
print(int(rang20[5]))
print(rang20)