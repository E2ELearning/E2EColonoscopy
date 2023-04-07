# A deep learning algorithm based on 1DCNN-LSTM for automatic sleep staging
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



gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        curLR = optimizer._decayed_lr(tf.float32)
        return curLR 
    return lr

def draw(Tloss,Vloss,ll,b,path1):  
    for i in range(len(Tloss)):     
        plt.figure()
        colours=['r','g','b','c','y','r']
        plt.plot(Tloss[i],colours[i])
        plt.plot(Vloss[i],color="black",linestyle = 'dashed')
        # Show/save figure as desired.
        plt.legend(['train', 'val_loss'], loc='upper right',fontsize=12,frameon=False)
        #plt.title('Train and validation Loss of batch size '+ str(n_params[i]),fontdict={'fontsize':14})
        plt.title('Train and Val Loss Lr '+ str(ll[i]) + " Bz " + str(b),fontdict={'fontsize':14})
        img='_{}_{}_{}.png'.format(i,b,ll[i])
        path11=os.path.join(path1,img)
        plt.savefig(path11,dpi=600)   
        #plt.show()

### Reservoir Production Prediction Model Based on a Stacked LSTM Network and Transfer Learning

# The
# number of hidden units in the four layers of the LSTM
# network is (50,50,50,50), and Adam and MSE are selected as
# the optimizer and loss function, respectively




### Remaining Useful Life Estimation in Prognostics 
#Using Deep Bidirectional LSTM Neural Network
def regression_BiLSTM(timestep):
    dropR=0.2
    INp = keras.Input(shape=(timestep,2))
    x = Bidirectional(LSTM(64,return_sequences=True))(INp)
    x = Bidirectional(LSTM(32,return_sequences=True))(x)
    #x= TimeDistributed(dense(25))(x)
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation(activation='relu')(x)
    x = Dense(8)(x)
    x = Activation(activation='relu')(x)
    x = Dense(2)(x)
    x = Activation(activation='linear')(x)
    output = Model(INp , x, name='regression_1Dcnn')
    return output





def Runall(trainX_, trainy_, testX_, testy_, Vax_, Vay_, epo, b,lr,n_timesteps,n_train,n_va,PATH):
    batch_size = b
    epochs = epo
    steps_per_epochss = ceil(n_train//b) # number of batchs for train
    validation_stepss=ceil(n_va//b) # number of the validation data's  batchs for validation
    
    model=regression_BiLSTM(n_timesteps)
    
    
    #Ad =keras.optimizer_v2.adam.Adam(learning_rate=lr,beta_1=0.9, beta_2=0.999, epsilon=1e-07,epsilon=None, decay=0.01, amsgrad=False)  
    Ad =Adam(learning_rate=lr,beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    decay_rates = [1E-1, 1E-2, 1E-3, 1E-4]
    opti=SGD(learning_rate=lr, momentum=0.9, decay = 0.001 )
    lr_metric = get_lr_metric(opti)
    model.compile(optimizer = Ad, loss = 'mean_squared_error', metrics = ['mae'])
    
    
    path=PATH
    #var='1DCNN_Epoch_{epoch:02d}_valloss_{val_loss:.3f}.h5'
    var='2BiLSTM.h5'
    filepath=os.path.join(path,'model',var)
    #filepath ='1DCNN_{epoch:02d}_{val_loss:.3f}.h5'
    keras_callbacks= [EarlyStopping(monitor='val_loss',  patience=100,baseline=None, mode='auto',restore_best_weights=True),
                  ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_freq='epoch', mode='min')]
    
    #model.save('1DCNN_Epoch_{epoch:02d}_valloss_{val_loss:.3f}.h5')
    history = model.fit(trainX_, trainy_, steps_per_epoch=steps_per_epochss,
                        validation_data=(Vax_,Vay_), 
                        validation_steps=validation_stepss,
                        epochs=epochs, 
                        verbose=0,
                        callbacks=keras_callbacks)
    score = model.evaluate(Vax_,Vay_, 
                           batch_size = batch_size, 
                           verbose=0)
    
    lossMSE = score[0]
    lossMAE = score[1]  
    TrainL= (history.history['loss'])
    ValL= (history.history['val_loss'])
    save(os.path.join(path,'model','TrainL{}_{}.npy'.format(batch_size,lr)),TrainL)
    save(os.path.join(path,'model','ValL{}_{}.npy'.format(batch_size,lr)),ValL)
    predictsave = []
    predictsave = model.predict(testX_)
    return lossMSE ,lossMAE ,TrainL, ValL, predictsave


root_folderxy=r'/home/rc/Desktop/syfile/1DCNN_Pre/Dataset'
all_X_data = sorted(os.listdir(root_folderxy))
print(all_X_data)
train_data_list, test_data_list = train_test_split(all_X_data, test_size=0.2, random_state = 0)
print("Train: ",train_data_list)
print("Test: ",test_data_list)

filename = sorted([os.path.join(root_folderxy, x) for x in ('data1.csv','data3.csv','data5.csv','data6.csv',
                                                            'data7.csv','data8.csv','data10.csv','data11.csv',
                                                            'data12.csv')])
filename_validation = sorted([os.path.join(root_folderxy, x) for x in ('data4.csv','data9.csv','data2.csv')])

print("------------------------------------")
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
print("Train: ",count,type(count))

countV=count_number_data(filename_validation)
countV=np.array(countV)
print("Validation ",countV,type(countV))

print("------------------------------------")

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
        print("dodai", dodai)
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
print(Xpix.shape)
Ysig=np.stack((VeloscaleX ,VeloscaleY), axis=1)
print(Ysig.shape)

XpixVa=np.stack((PxVa,PyVa), axis=1)
print(XpixVa.shape)
YsigVa=np.stack((VeloscaleXVa ,VeloscaleYVa), axis=1)
print(YsigVa.shape)

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



step=40
ep=100
n_params = [64,128,256]
learing = [0.001,0.0001]


TrainP1,TrainS1,dem1 = Readtrain(Xpix,Ysig,step,count)
TrainP1 = np.array(TrainP1,dtype=np.float32)  
TrainS1 = np.array(TrainS1,dtype=np.float32) 
# print("train pixel",TrainP1.shape,TrainP1.dtype)
# print("train signal",TrainS1 .shape,TrainS1 .dtype)


ValidationX,ValidationY ,n = Readtrain(XpixVa,YsigVa,step,countV)

ValidationX = np.array(ValidationX,dtype=np.float32)  
ValidationY  = np.array(ValidationY ,dtype=np.float32) 
# print("Validation pixel",ValidationX.shape,ValidationX.dtype)
# print("Validation signal",ValidationY.shape,ValidationY.dtype)





##only signal form experiment 
trainX, trainy = TrainP1, TrainS1
testX, testy = ValidationX[1363:2363,:],ValidationY[1363:2363,:]
# print("Testset X ",testX.shape,testX.dtype)
# print("Testset Y ",testy.shape,testy.dtype)
row,col=testy.shape
datapoint = np.arange(0, row)
Ytest = testy
scores = list()

n_train,_,_= trainX.shape
print("number of train",n_train)
n_valid,_,_= ValidationX.shape
print("number of validation",n_valid)
n_test,_,_= testX.shape
print("number of test",n_test)

print("=========== Start Train ==============")
   
        
for r in range(len(n_params)):
    print("")
    print(" ____________ bachsize _____________ ",n_params[r])
    print("")
    Ba=n_params[r]
    TRAINloss = []
    VALloss =[]
    VALloss =[]
    Pre = []
    for k in range(len(learing)):  
        save_figure_path='./2BiLSTM_{}_{}_{}_{}'.format(Ba,learing[k],ep,step)
        title = "./Prediction_{}_{}.png".format(Ba,learing[k])
        #save_figure_path = os.path.join(save_figure_path, 'predictions', 'model')
        save_figure_path = os.path.join(save_figure_path, 'predictions')
        os.makedirs(save_figure_path, exist_ok=True)
        
     
        print("         bachsize __ ",Ba," ++ learning rate ++ ", learing[k] )
        Lmse,Lmae,Tl,Vl,p  = Runall(trainX, trainy, testX, testy,ValidationX,ValidationY,ep,Ba,learing[k],step,n_train,n_valid,save_figure_path)
        print(" Loss MSE --------------------------------------------------",np.round(Lmse,5))
        #print('>#%d: %.3f' % (r, Lmse))
        # print(" Loss MAE ")
        # print('>#%d: %.3f' % (r, Lmae))
        TRAINloss.append(Tl)
        VALloss.append(Vl)
        scores.append(Lmse)
        Pre.append(p)

        fig, axs = plt.subplots(2,figsize=(12,5))
        fig.suptitle('Prediction from Test set',fontsize=16)
        axs[0].plot(datapoint,p[:,0],color='red',label = "X Prediction")
        axs[0].plot(datapoint,Ytest[:,0],color='black',label = "X truth") 
        axs[0].set_title('X prediction',fontsize=14)
        axs[0].set_xlabel('Number of datapoint', fontsize=12)
        axs[0].set_ylabel('X prediction', fontsize=12)                 
        axs[0].legend(['X Prediction', 'X truth'],loc="upper left",prop={"size":12},frameon=False)

        axs[1].plot(datapoint,p[:,1],color='blue',label = "Y Prediction")
        axs[1].plot(datapoint,Ytest[:,1],color='black',label = "Y truth")
        axs[1].set_title(' Y prediction',size=14)
        axs[1].set_xlabel('Number of datapoint', fontsize=12)
        axs[1].set_ylabel('Y prediction', fontsize=12)
        axs[1].legend(['Y Prediction', 'Y truth'],loc="upper left",prop={"size":12},frameon=False)
        
        fig.tight_layout() 
        
        save(os.path.join(save_figure_path ,'TestPrediction_{}_{}_{}_{}.npy'.format(Ba,learing[k],ep,step)),p)
        save(os.path.join(save_figure_path ,'TestGroundTruth_{}_{}_{}_{}.npy'.format(Ba,learing[k],ep,step)),Ytest)
        # save_figure_path='./saved_model{}_{}'.format(Ba,learing[k])
        # title = "./Prediction_{}_{}.png".format(Ba,learing[k])
        # save_figure_path = os.path.join(save_figure_path, 'predictions', 'figures')
        # os.makedirs(save_figure_path, exist_ok=True)
        # path=os.path.join(save_figure_path,title)
        
        path=os.path.join(save_figure_path,'{}_{}.png'.format(Ba,learing[k]))
        plt.savefig(path, dpi=600)  
         
        #plt.show()  
        
    draw(TRAINloss,VALloss,learing,Ba,save_figure_path )

