from cv2 import mean
import numpy as np
import itertools
import matplotlib.pyplot as plt
import math
trainloss=np.load(r'C:\Users\217\Desktop\Sy_donot_delete\PT202208\saved_modelCNN128lr0d00001\loss\CRNN_epoch_training_loss.npy',allow_pickle='TRUE')
validloss=np.load(r'C:\Users\217\Desktop\Sy_donot_delete\PT202208\saved_modelCNN128lr0d00001\loss\CRNN_epoch_valid_loss.npy',allow_pickle='TRUE')
print(type(trainloss))
row=trainloss.shape

row=np.asarray(row)
datapoint=np.arange(0,row)
print(datapoint)

fig = plt.figure(figsize=(12, 5))
#fig = plt.figure(figsize=(8, 4))
plt.plot(datapoint,trainloss,color='blue',marker='o',markersize=5)
plt.plot(datapoint,validloss,color='green',marker='*',markersize=5)

# plt.plot(datapoint,trainloss,color='blue',markersize=5)
# plt.plot(datapoint,validloss,color='green',markersize=5)
new_list = range(math.floor(min(datapoint)), math.ceil(max(datapoint))+1)
plt.xticks(new_list)
plt.title('Train and validation Loss',fontdict={'fontsize':14})
plt.ylabel('val_loss',fontsize=12)
plt.xlabel('epoch',fontsize=12)
plt.legend(['Train_loss', 'Val_loss'], loc='upper right',fontsize=12,frameon=False)
fig.tight_layout()
plt.show()