
### pytorchEnv 
import os
import pandas as pd
import numpy as np
from numpy import save
from numpy import load
import matplotlib.pyplot as plt

# def Train_Val(train,val):
#     T=np.load(train)
#     V=np.load(val)
#     epochs=len(T)
#     fig = plt.figure(figsize=(8, 4))
#     plt.plot(np.arange(1, epochs + 1), T)  
#     plt.plot(np.arange(1, epochs + 1), V,color='#e35f62',linestyle = 'dashed') 
#     plt.title("Model loss")
#     plt.xlabel('Epochs',fontsize=12)
#     plt.ylabel('Loss',fontsize=12)
#     plt.legend(['Train', 'Validation'], loc='upper right',fontsize=12,frameon=False)
#     title = "1DCNN64_1.png"
#     plt.savefig(title, dpi=600)


#train1=r'F:\ex90\90\1DCNN_Pre\1DCNN_64_0.0001_100_40\predictions\model\TrainL64_0.0001.npy'
Validate1=r'C:\Users\217\Desktop\Sy_donot_delete\LV20221117\res1834ubuntune\res34bz64lr5\loss\CRNN_epoch_valid_loss.npy'
#T=np.load(train1)
V=np.load(Validate1)
print(min(V))

