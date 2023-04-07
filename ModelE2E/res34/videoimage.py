

import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import os
from PIL import Image

img_array1 = []
img_array2 = []
exp1 = '/home/rc/Desktop/syfile2023/Thi8'
jpg = '/*'

for filename1 in sorted(glob.glob(exp1+jpg), key=os.path.getctime):
    img = cv2.imread(filename1)
    height, width, layers = img.shape
    size = (width,height)
    img_array1.append(img)
    #print(filename1)

img_array1=np.array(img_array1)
print(img_array1.shape)
pre=np.load('/home/rc/Desktop/syfile2023/regrression4variables/Y_preThi8.npy')
tar=np.load('/home/rc/Desktop/syfile2023/regrression4variables/Y_targetThi8.npy')
print(tar.shape)
number,_,_=tar.shape
print(number)
count=1234
codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter('{}.avi'.format('nguyenvansy1'),cv2.VideoWriter_fourcc(*'DIVX'),fps=15, frameSize=(1640,480))
out1 = cv2.VideoWriter('{}.avi'.format('ImageThi8'),codec,fps=15, frameSize=(640,480))
out2 = cv2.VideoWriter('{}.avi'.format('targetThi81'),codec,fps=15, frameSize=(1000,480))
exp2 = '/home/rc/Desktop/syfile2023/saveThi8'
for file1,file2 in zip(sorted(glob.glob(exp1+jpg)),sorted(glob.glob(exp2+jpg))):
    #print(file1)
    #print(file2)
    # print("--------------")
    img1 = cv2.imread(file1)
    img2 = cv2.imread(file2)
    
    # height, width, layers = img1.shape
    # size = (width,height)
    img2=cv2.resize(img2, (1000,480))
    #print(img1.shape)
    #print(img2.shape)
    img_3 = np.concatenate((img1,img2), axis=1)
    print(img_3.shape)
    # cv2.imshow('frame',img_3 )
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    height, width, layers = img_3.shape
    print(height, width)
    out.write(img_3)
    
    #height, width, layers = img2.shape
    #print(height, width)
    #out1.write(img1)
    #out2.write(img2)

cv2.destroyAllWindows()
out2.release()








# my_path='/home/rc/Desktop/syfile2023/saveThi81/'

# for i in range(number):
#     #print(i)
#     fig1 = plt.figure(figsize=(12,6))
#     a="crimson"
#     b="seagreen"
#     plt.subplot(4, 1, 1)
#     plt.plot(tar[:i,:,0],color=a)
#     plt.plot(pre[:i,:,0],color=b)
#     plt.legend(['target', 'pred'], loc="upper right")
#     plt.xlim([-10,number + 50])
#     plt.ylim([-1.1,1.1])
#     plt.xlabel('frame')
#     plt.ylabel('steer X')

#     plt.subplot(4, 1, 2)   
#     plt.plot(tar[:i,:,1],color=a)
#     plt.plot(pre[:i,:,1],color=b)
#     plt.legend(['target', 'pred'], loc="upper right")
#     plt.xlim([-10,number + 50])
#     plt.ylim([-1.1,1.1])
#     plt.xlabel('frame')
#     plt.ylabel('steer Y')

#     plt.subplot(4, 1, 3)   
#     plt.plot(tar[:i,:,2],color=a)
#     plt.plot(pre[:i,:,2],color=b)
#     plt.legend(['target', 'pred'], loc="upper right")
#     plt.xlim([-10,number + 50])
#     plt.ylim([-0.2,1.1])
#     plt.xlabel('frame')
#     plt.ylabel('Feeding velocity')

#     plt.subplot(4, 1, 4)   
#     plt.plot(tar[:i,:,3],color=a)
#     plt.plot(pre[:i,:,3],color=b)
#     plt.legend(['target', 'pred'], loc="upper right")
#     plt.xlim([-10,number + 50])
#     plt.ylim([-0.2,1.1])
#     plt.xlabel('frame')
#     plt.ylabel('Collision')
#     plt.tight_layout()
#     print(count)
#     plt.savefig(my_path+'books_read{}.png'.format(count),dpi=100)
#     count = count+1
    
 
