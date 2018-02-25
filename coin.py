import skimage.morphology as sm
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure,draw
from numpy import *
from skimage import io
x=io.imread('C:/Users/Administrator/Desktop/py/coin.jpg',as_grey=True)#打开硬币图片
#io.imshow(x)


i,j=x.shape
bg=zeros([i,j])
y=zeros([i,j])
for m in range(i):
    for n in range(j):
        if x[m,n]<0.5:
            bg[m,n]=x[m,n]
        else:
            bg[m,n]=0
y=x-bg
#io.imshow(bg)
#io.imshow(y)

yy=zeros([i,j])
for m in range(i):
    for n in range(j):
        if y[m,n] >=0.2:
            yy[m,n]=1
        else:
            yy[m,n]=0
#io.imshow(yy)
yy=sm.erosion(yy,sm.square(3))    #腐蚀 系数可以适当调整
#io.imshow(yy)

import cv2
kernel = np.ones((3,3),np.uint8)
yyy = cv2.dilate(yy,kernel,iterations =1)  #膨胀  系数可以适当调整
#io.imshow(yyy)

contours = measure.find_contours(yyy, 0.9)  #得到一系列边界（x，y）
numcoin=len(contours)                       #边界个数就是硬币个数
print('共有%d个硬币'%(len(contours)))          





yyyy=zeros([i,j])
for m in range(i):
    for n in range(j):
        if yyy[m,n]==1:
            yyyy[m,n]=x[m,n]
        else:
            yyyy[m,n]=0




from matplotlib import pyplot as plt
#%matplotlib qt5
from skimage import io
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']   #可以使用中文标签


subplot(221)
io.imshow(x)
axis('off')
title('原图')

subplot(222)
io.imshow(y)
axis('off')
title('去除背景')

subplot(223)
io.imshow(yyy)
axis('off')
title('腐蚀与膨胀操作')

subplot(224)
io.imshow(yyyy)
axis('off')
title('有'+str(numcoin)+'个硬币')
plt.tight_layout()
