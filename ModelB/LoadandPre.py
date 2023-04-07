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
from tensorflow.keras.layers import Conv1D ,AveragePooling1D
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
from sklearn.metrics import mean_squared_error



gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# def loadmodel(pa,ValidationX,ValidationY):
#     model1=tf.keras.models.load_model(pa,compile=False)
#     print("Validationx  ",ValidationX.shape)
#     print("Validationy  ",ValidationY.shape)
#     number_point,_,_=ValidationX.shape
#     print("number_point",number_point)
#     result=[]
#     for i in range(number_point):
#         ad=np.expand_dims(ValidationX[i], axis=0)
#         predict = model1.predict(ad)
#         result.append(predict)
        
#     result=np.array(result,dtype=np.float32) 
#     print("result :",result.shape)
#     result=np.squeeze(result)
#     print("result :",result.shape)
#     print("target:",ValidationY.shape)
#     return

root_folderxy=r'/home/rc/Desktop/syfile/1DCNN_Pre/Dataset'
all_X_data = sorted(os.listdir(root_folderxy))
#print(all_X_data)
train_data_list, test_data_list = train_test_split(all_X_data, test_size=0.2, random_state = 0)
#print("Train: ",train_data_list)
#print("Test: ",test_data_list)

filename = sorted([os.path.join(root_folderxy, x) for x in ('data1.csv','data3.csv','data5.csv','data6.csv',
                                                            'data7.csv','data8.csv','data10.csv','data11.csv',
                                                            'data12.csv')])
filename_validation = sorted([os.path.join(root_folderxy, x) for x in ('data4.csv','data9.csv','data2.csv')])

#print("------------------------------------")
def count_number_data(CSV):
    number = []
    row = 0
    for l in range(len(CSV)):
        p = pd.read_csv(CSV[l])
        row = len(p)
        number.append(row)
    return number


count=count_number_data(filename)
count=np.array(count)
#print("Train: ",count,type(count))

countV=count_number_data(filename_validation)
countV=np.array(countV)
#print("Validation ",countV,type(countV))

#print("------------------------------------")

def readpixelandsignal(lis_of_file):
    PiX=[]
    PiY=[]
    vel_x=[]
    vel_y=[]
    for data in range(len(lis_of_file)):
        dfX = pd.read_csv(lis_of_file[data], skipinitialspace=True, usecols=["deX","deY","VeX","VeY"],doublequote=False)
        piXpre=dfX['deX']
        piYpre=dfX['deY']
        vex=dfX['VeX'].str.strip('[]').astype(float)
        vey=dfX['VeY'].str.strip('[]').astype(float)
        
        piXpre=np.asarray(piXpre)
        piYpre=np.asarray(piYpre)
        vex=np.asarray(vex)
        vey=np.asarray(vey)
        
        dodai=piXpre.shape[0]
        #print("dodai", dodai)
        for i in range(dodai):
            a = min(max(vex[i],-15),15)
            b = min(max(vey[i],-15),15)
            
            vel_x.append(a)
            vel_y.append(b)
            PiX.append(piXpre[i])
            PiY.append(piYpre[i])    
    return vel_x,vel_y,PiX,PiY

Vx,Vy,Px,Py=readpixelandsignal(filename)
VxVa,VyVa,PxVa,PyVa=readpixelandsignal(filename_validation)

Vx=np.array(Vx,dtype=np.float32)
Vy=np.array(Vy,dtype=np.float32)
Px=np.array(Px,dtype=np.float32)
Py=np.array(Py,dtype=np.float32)

VxVa=np.array(VxVa,dtype=np.float32)
VyVa=np.array(VyVa,dtype=np.float32)
PxVa=np.array(PxVa,dtype=np.float32)
PyVa=np.array(PyVa,dtype=np.float32)


# print("Vx",Vx.shape, "Vy",Vy.shape,"Px",Px.shape,"Px",Px.shape)
# print(" max min volocity", max(Vx),min(Vx))
# print(" max min volocity",max(Vy),min(Vy))
# print(" max min Trainsignal", max(Px),min(Px))
# print(" max min Trainsignal",max(Py),min(Py))
# print("------------------------------------")
# print(" ")
rang1=np.linspace(-15,15,1000)
rang2=np.linspace(-1,1,1000)
VeloscaleX =np.interp(Vx,rang1,rang2)
VeloscaleY =np.interp(Vy,rang1,rang2)
Px[Px > 1] = 1
Px[Px < -1] = -1
Py[Py > 1] = 1
Py[Py < -1] = -1

# print("VxV",VxVa.shape, "VyV",VyVa.shape,"Px",PxVa.shape,"Px",PxVa.shape)
# print(" max min volocity validation", max(VxVa),min(VxVa))
# print(" max min volocity validation",max(VyVa),min(VyVa))
# print(" max min Trainsignal validation", max(PxVa),min(PxVa))
# print(" max min Trainsignal validation",max(PyVa),min(PyVa))

# print("------------------------------------")
# print(" ")
VeloscaleXVa =np.interp(VxVa,rang1,rang2)
VeloscaleYVa =np.interp(VyVa,rang1,rang2)
PxVa[PxVa > 1] = 1
PxVa[PxVa < -1] = -1
PyVa[PyVa > 1] = 1
PyVa[PyVa < -1] = -1

# print(" max min volocity", max(VeloscaleX),min(VeloscaleX))
# print(" max min volocity",max(VeloscaleY),min(VeloscaleY))
# print(" max min Trainsignal", max(Px),min(Px))
# print(" max min Trainsignal",max(Py),min(Py))
# print("------------------------------------")
# print(" ")


Xpix=np.stack((Px,Py), axis=1)
#print(Xpix.shape)
Ysig=np.stack((VeloscaleX ,VeloscaleY), axis=1)
#print(Ysig.shape)

XpixVa=np.stack((PxVa,PyVa), axis=1)
#print(XpixVa.shape)
YsigVa=np.stack((VeloscaleXVa ,VeloscaleYVa), axis=1)
#print(YsigVa.shape)

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
        #print("       plus step     ",k)
        number.append(dem) 
    #print("  l ",l)
    return pix,sig,number


step=40
ep=30
n_params = [64,128,256]
learing = [0.001,0.0001]


TrainP1,TrainS1,dem1 = Readtrain(Xpix,Ysig,step,count)
TrainP1 = np.array(TrainP1,dtype=np.float32)  
TrainS1 = np.array(TrainS1,dtype=np.float32) 
#print("train pixel",TrainP1.shape,TrainP1.dtype)
#print("train signal",TrainS1 .shape,TrainS1 .dtype)


ValidationX,ValidationY ,n = Readtrain(XpixVa,YsigVa,step,countV)

ValidationX = np.array(ValidationX,dtype=np.float32)  
ValidationY  = np.array(ValidationY ,dtype=np.float32) 
#print("Validation pixel",ValidationX.shape,ValidationX.dtype)
#print("Validation signal",ValidationY.shape,ValidationY.dtype)





##only signal form experiment 
trainX, trainy = TrainP1, TrainS1
testX, testy = ValidationX[1363:2363,:],ValidationY[1363:2363,:]
#print("Testset X ",testX.shape,testX.dtype)
#print("Testset Y ",testy.shape,testy.dtype)
row,col=testy.shape
datapoint = np.arange(0, row)
Ytest = testy
scores = list()

n_train,_,_= trainX.shape
#print("number of train",n_train)
n_valid,_,_= ValidationX.shape
#print("number of validation",n_valid)
n_test,_,_= testX.shape
#print("number of test",n_test)

#print("=========== Start Train ==============")

model1=tf.keras.models.load_model(r'/home/rc/Desktop/syfile/1DCNN_Pre/DCNNLSTMpoeposed12022112patien100_256_0.001_100_40/predictions/model/Pro.h5',compile=False)
print("Validationx  ",ValidationX.shape)
print("Validationy  ",ValidationY.shape)
number_point,_,_=ValidationX.shape
print("number_point",number_point)
result=[]
for i in range(number_point):
    ad=np.expand_dims(ValidationX[i], axis=0)
    predict = model1.predict(ad)
    result.append(predict)

result=np.array(result,dtype=np.float32) 
print("result :",result.shape)
result=np.squeeze(result)

print("result :",result.shape)
print("target:",ValidationY.shape)

from sklearn.metrics import mean_squared_error

MSEall = mean_squared_error(ValidationY,result)

print(MSEall)

## 0.020659398  - 0.02164
##  0.057355836  - 0.05936