

import matplotlib
from sklearn import metrics
#from DenseResidual1D import X
#matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import multiprocessing
import time
import random
import os
import cv2 
import glob
import pandas as pd
import numpy as np
import pickle
from scipy.interpolate import interp1d
import csv 
#from skimage import io, transform
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import ceil
from sklearn.metrics import r2_score ,mean_squared_error
### Get Reproducible Results with Keras #####
from numpy.random import seed
seed(42)
np.random.seed(2)
from numpy import save
from numpy import load

root_folder=r'/home/rc/Desktop/syfile2023/Thesisdata2023_newChangeName/labels'
filenameexperiment = sorted([os.path.join(root_folder, x) for x in ('Thi1.csv','Thi2.csv','Thi3.csv','Thi4.csv','Thi5.csv',
                                                                    'Thi6.csv','Thi7.csv','Thi8.csv','Thi9.csv','Thi10.csv',
                                                                    'Thi12.csv','Thi13.csv','Thi14.csv','Thi15.csv',
                                                                    'Thi16.csv','Thi17.csv','Thi18.csv','Thi19.csv','Thi20.csv',
                                                                    'Thi22.csv','Thi23.csv','Thi24.csv','Thi25.csv',
                                                                    'Thi26.csv','Thi27.csv','Thi28.csv','Thi29.csv','Thi30.csv',
                                                                    'Thi31.csv','Thi32.csv')])

def function1(lis_of_file):
    A=[]
    B=[]
    C=[]
    D=[]
    for data in range(len(lis_of_file)):
        dfX = pd.read_csv(lis_of_file[data], skipinitialspace=True, usecols=["VeX","VeY","coll","Feedv"],doublequote=False)
        Vx=dfX['VeX']
        Vy=dfX['VeY']
        collision=dfX['coll']
        Feeding=dfX['Feedv']
        #print(Vx.shape,Vy.shape,collision.shape,Feeding.shape)
        
        Vx=np.asarray(Vx).astype(float)
        Vy=np.asarray(Vy).astype(float)
        collision=np.asarray(collision).astype(float)
        Feeding=np.asarray(Feeding).astype(float)
        
        #print(Vx.shape)
        number=len(Vx)
        #print(number)
        for i in range(number):
            #print(Vx[i],Vy[i],collision[i],Feeding[i])
            #print(type(Vx[i]),type(Vy[i]),type(collision[i]),type(Feeding[i]))
            A.append(Vx[i])
            B.append(Vy[i])
            C.append(collision[i])
            D.append(Feeding[i])
        
    return A,B,C,D

VEx,VEy,Coll,F=function1(filenameexperiment)

VEx=np.array(VEx)
VEy=np.array(VEy)
Coll=np.array(Coll)
F=np.array(F)


print(VEx.shape,type(VEx))
K=len(VEx)
print(K)

print(max(F),min(F)) #63
normalizedF = (F-min(F))/(max(F)-min(F))

# plt.hist(VEx, bins=100,color='g')
# plt.hist(VEx, bins=100,color='r')
#plt.hist(Coll, bins=100,color='b')
#plt.hist(F, bins=100,color='royalblue')
plt.hist(normalizedF, bins=100,color='royalblue')
plt.xlabel("Velocity in X direction")
plt.ylabel("Frequency")
plt.title("Histogram of VEx")
plt.show()