import numpy as np

t = np.load('./cnnlstm64_5/loss/CRNN_epoch_training_loss.npy')
v = np.load('./cnnlstm64_5/loss/CRNN_epoch_valid_loss.npy')

ind = np.argmin(v)
print('{:.4e}, {:.4e}'.format(t[ind], v[ind]))
print(min(v))