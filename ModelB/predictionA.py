
from sklearn import metrics
import multiprocessing
import time
import random
import os
import cv2 
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python import keras
from scipy.interpolate import interp1d
import csv 
import matplotlib.pyplot as plt
from numpy import save
from numpy import load
import re
from scipy.interpolate import interp1d
from tensorflow.keras.models import Sequential, Model, load_model, save_model

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


    
#model4_256
#model1= tf.keras.models.load_model(r'/home/rc/Desktop/syfile/Synew/1DCNN_Epoch_09_valloss_0.137.h5',compile=False )


#model6_1
#model1= tf.keras.models.load_model(r'/home/rc/catkin_ws/src/predic1/src/1DCNN_Epoch_15_valloss_0.136.h5',compile=False)
#resnetCNNLSTM
#model2= tf.keras.models.load_model(r'/home/rc/catkin_ws/src/predic1/src/1DCNN_Epoch_40_valloss_0.166.h5',compile=False) 
 
model1= tf.keras.models.load_model(r'/home/rc/Desktop/syfile/1DCNN_Pre/DCNNLSTMpoeposed12022112patien100_128_0.001_100_40/predictions/model/Pro.h5',compile=False )
  
dfX = pd.read_csv(r"/home/rc/catkin_ws/src/save1/E3/data.csv", skipinitialspace=True, usecols=["deX","deY","VeX","VeY"],doublequote=False)
piXpre=dfX['deX']
piYpre=dfX['deY']
vex=dfX['VeX'].str.strip('[]').astype(float)

vey=dfX['VeY'].str.strip('[]').astype(float)

piXpre=np.asarray(piXpre)
piYpre=np.asarray(piYpre)
vex=np.asarray(vex)
vey=np.asarray(vey)



print("min velocity X ",min(vex) ," max velocity X ", max(vex))
print("min velocity Y ",min(vey) ," max velocity Y ", max(vey))

vel_x=[]
vel_y=[]
for i in range(len(vex)):
    a = min(max(vex[i],-15),15)
    b = min(max(vey[i],-15),15)
    vel_x.append(a)
    vel_y.append(b)
    
vel_x=np.asarray(vel_x)
vel_y=np.asarray(vel_y)
print("min velocity X after",min(vel_x) ," max velocity X after ", max(vel_x))
print("min velocity Y after ",min(vel_y) ," max velocity Y  after", max(vel_y))
print(vex.shape)
print(vel_x.shape)


rang1=np.linspace(-15,15,1000)
rang2=np.linspace(-1,1,1000)

VeloscaleX =np.interp(vel_x,rang1,rang2)
#print(VeloscaleX)
VeloscaleY =np.interp(vel_y,rang1,rang2)

step=40
Xpix=np.stack((piXpre,piYpre), axis=1)
print(Xpix.shape)
Ysig=np.stack((VeloscaleX ,VeloscaleY), axis=1)
print(Ysig.shape)



n_test,_=Xpix.shape
pixalEP = []
veloEP =[]
for j in range(n_test-step+1):
    e = j+step
    pixalEP.append(Xpix[j:e,])
    veloEP.append(Ysig[e-1,])

pixalEP = np.array(pixalEP,dtype=np.float32)  
veloEP= np.array(veloEP,dtype=np.float32) 
print(pixalEP.shape,pixalEP.dtype)
print(veloEP.shape,veloEP.dtype)




number_point,_,_=pixalEP.shape
print(number_point)
result=[]
for i in range(number_point):
    ad=np.expand_dims(pixalEP[i], axis=0)
    predict = model1.predict(ad)
    #print(predict)
    #print("")
    result.append(predict)
    
    

result=np.array(result,dtype=np.float32) 
result=np.squeeze(result)


datapoint = np.arange(0,number_point)
print(result.shape)
print(veloEP.shape)

#[]

DFtest = pd.read_csv(r"/home/rc/Desktop/syfile/sy/DataframeTest.csv", skipinitialspace=True, usecols=["PixX","PixY","sx","sy"])
piXtest=DFtest ['PixX']
piYtest=DFtest ['PixY']
sxtest=DFtest ['sx']
sytest=DFtest ['sy']

piXtest=np.asarray(piXtest)
piYtest=np.asarray(piYtest)
Sxtest=np.asarray(sxtest)
Sytest=np.asarray(sytest)

ValPX=np.stack((piXtest,piYtest), axis=1)
print(ValPX.shape)
ValSig=np.stack((Sxtest,Sytest), axis=1)
print(ValSig.shape)

ValPX[:,0] = 1-(2*ValPX[:,0])
ValPX[:,1] = 1-(2*ValPX[:,1])

n_test,_= ValPX.shape
print("number of test",n_test)

B=[1741, 1571, 1762]
def Readtrain(Pixel,Singal,step,matrix):
    sig=[]
    pix=[]
    number=[]
    A=matrix
    l=0
    d=0
    k=0
    for data in range(len(A)):
        dem=0
        dodai=A[data]
        for i in range(dodai-step+1):
            d = k + step
            pix.append(Pixel[k:d])
            sig.append(Singal[d-1])
            k += 1
            dem += 1
            l+=1 

        k += step 
        k = k -1
        print("       plus step     ",k)
        number.append(dem) 
        
    print("  l ",l)
        
    return pix,sig,number


ValP1,valS1,dem12= Readtrain(ValPX,ValSig,step,B)

ValP1 = np.array(ValP1,dtype=np.float32)  
valS1= np.array(valS1,dtype=np.float32) 
print("train pixel",ValP1.shape,ValP1.dtype)
print("train signal",valS1.shape,valS1.dtype)

testX, testy = ValP1[:1700],valS1[:1700]
row,col=testy.shape
print(row)
datapoin_test = np.arange(0, row)

result_test=[]
for i in range(row):
    ad_test=np.expand_dims(testX[i], axis=0)
    predict_test = model1.predict(ad_test)
    #print(predict)
    #print("")
    result_test.append(predict_test)


result_test=np.array(result_test,dtype=np.float32) 
result_test=np.squeeze(result_test)
print(result_test.shape)

#save('result.npy', result)
#save('veloEP.npy', veloEP)

## drawwwwwwwwwwwww result
fig, axs = plt.subplots(2,figsize=(12,5))
fig.suptitle('Prediction from Experiment',fontsize=16)
axs[0].plot(datapoint ,result[:,0],color='red',label = "X Prediction of model")
axs[0].plot(datapoint ,veloEP[:,0],color='black',label = "X Prediction of experiment")
axs[0].set_title('Prediction',fontsize=14)
axs[0].set_xlabel('Number of datapoint', fontsize=12)
axs[0].set_ylabel('X prediction', fontsize=12)                 
axs[0].legend(['X Prediction', 'X Exper -velocity'],loc="upper left",prop={"size":12},frameon=False)

axs[1].plot(datapoint ,result[:,1],color='blue',label = "Y Prediction of model")
axs[1].plot(datapoint ,veloEP[:,1],color='black',label = "Y Prediction of experiment")
axs[1].set_title(' Y prediction',size=14)
axs[1].set_xlabel('Number of datapoint', fontsize=12)
axs[1].set_ylabel('Y prediction', fontsize=12)
axs[1].legend(['Y Prediction', 'Y Exper-velocity'],loc="upper left",prop={"size":12},frameon=False)
fig.tight_layout()



fig1, axs1 = plt.subplots(2,figsize=(12,5))
fig1.suptitle('Prediction from Test set',fontsize=16)
axs1[0].plot(datapoin_test,result_test[:,0],color='red',label = "X Prediction")
axs1[0].plot(datapoin_test,testy[:,0],color='black',label = "X truth")
axs1[0].set_title('Prediction',fontsize=14)
axs1[0].set_xlabel('Number of datapoint', fontsize=12)
axs1[0].set_ylabel('X prediction', fontsize=12)                 
axs1[0].legend(['X Prediction', 'X truth'],loc="upper left",prop={"size":12},frameon=False)

axs1[1].plot(datapoin_test,result_test[:,1],color='blue',label = "Y Prediction")
axs1[1].plot(datapoin_test,testy[:,1],color='black',label = "Y truth")
axs1[1].set_title(' Y prediction',size=14)
axs1[1].set_xlabel('Number of datapoint', fontsize=12)
axs1[1].set_ylabel('Y prediction', fontsize=12)
axs1[1].legend(['Y Prediction', 'Y truth'],loc="upper left",prop={"size":12},frameon=False)
fig1.tight_layout()

plt.show()

# import numpy as np
# history=np.load('/home/rc/Desktop/syfile/Synew/ResCNNSLTMMM128_0.001_100_40/predictions128_0.001_40_my_history.npy',allow_pickle='TRUE').item()
    
# #print(history.items()) #prints keys and values
# print(history.keys()) #prints keys
# #print(history.values()) #prints values
    
# value = history['loss']
# print(value)

