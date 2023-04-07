from cv2 import mean
import numpy as np
import itertools

from sklearn.metrics import f1_score, confusion_matrix, mean_absolute_percentage_error, explained_variance_score, r2_score, mean_absolute_error, mean_squared_error


reg_true = np.load('./reg_true.npy')
reg_pred = np.load('./reg_pred.npy')

print(reg_true[...,0], reg_pred[...,0])

reg_true_x = list(itertools.chain(*reg_true[...,0]))
reg_true_y = list(itertools.chain(*reg_true[...,1]))
reg_pred_x = list(itertools.chain(*reg_pred[...,0]))
reg_pred_y = list(itertools.chain(*reg_pred[...,1]))


mape = mean_absolute_percentage_error(reg_true_x, reg_pred_x)
ev = explained_variance_score(reg_true_x, reg_pred_x)
r2 = r2_score(reg_true_x, reg_pred_x)
mae = mean_absolute_error(reg_true_x, reg_pred_x)
mse = mean_squared_error(reg_true_x, reg_true_y)
rmse = mse**0.5
print(mse)


mse = 0
for i ,j in zip(reg_true_x, reg_pred_x):
    mse += (i-j)**2

mse /= len(reg_true_x)

print(mse**0.5)

mse = 0
for i ,j in zip(reg_true_y, reg_pred_y):
    mse += (i-j)**2

mse /= len(reg_true_y)

print(mse**0.5)



# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(10,6))
# fig.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
# ax = fig.add_subplot(projection = '3d')

# start = 50
# end = 200

# y = list(np.arange(len(reg_true_x[start:end])))
# ax.plot(reg_true_x[start:end], y,reg_true_y[start:end])
# ax.plot(reg_pred_x[start:end], y,reg_pred_y[start:end])
# ax.set_xlim([0,1])
# ax.set_ylim([0,len(y)])
# ax.set_zlim([0,1])
# plt.show()