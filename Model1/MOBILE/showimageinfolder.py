import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy

images = []
for img_path in glob.glob(r'C:/Users/217/Desktop/Sy_donot_delete/LV20221117___________/Train_colonoscopy_mobilenetv2/lableandprediction/Sequence10/*.jpg'):
    images.append(mpimg.imread(img_path))

plt.figure(figsize=(20,10))
numberofimages = numpy.array(images)
print(type(numberofimages))
print(numberofimages)
num,_,_,_=numberofimages.shape

XY_groundTruth = numpy.load(r'C:\Users\217\Desktop\Sy_donot_delete\LV20221117___________\Train_colonoscopy_mobilenetv2\lableandprediction\all_y.npy')
XY_prediction = numpy.load(r'C:\Users\217\Desktop\Sy_donot_delete\LV20221117___________\Train_colonoscopy_mobilenetv2\lableandprediction\all_y_pred.npy')
Prob = numpy.load(r'C:\Users\217\Desktop\Sy_donot_delete\LV20221117___________\Train_colonoscopy_mobilenetv2\Seq10\probability.npy')

for i in range(0,num):
    i=i*3
    image=numberofimages[i]
    
    XY_g=XY_groundTruth[i]
    XY_g=XY_g[0,:]
    X_g=round(XY_g[0]*460)
    Y_g=round(XY_g[1]*480)
    print(X_g,Y_g)
    
    print("-----")
    XY_p=XY_prediction[i]
    XY_p=XY_p[0,:]
    X_p=round(XY_p[0]*460)
    Y_p=round(XY_p[1]*480)
    print(X_p,Y_p)
    pro=Prob[i]
    print(pro)
    print(image.shape)
    plt.text(100, -50,pro, bbox=dict(fill=False, edgecolor='red', linewidth=2))
    plt.scatter(X_g,Y_g, s=100,color="black")
    plt.scatter(X_p,Y_p, s=100,color="green",cmap='Greens',alpha=0.5)
    plt.imshow(image)
    plt.show()
    
    
print("aaa")