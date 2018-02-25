# coin2
一、计算硬币个数
二、计算硬币大小（像素）

1、图片素材
（使用python3.6, spyder开发环境，只能求分离的、色差明显的物体数量）
https://raw.githubusercontent.com/labAxiaoming/coin2/678adb94dd6f0eaf49314ed48bd8ae6a66f5f04a/0.jpg
https://raw.githubusercontent.com/labAxiaoming/coin2/678adb94dd6f0eaf49314ed48bd8ae6a66f5f04a/1.jpg


2、
python 打开图片，转为灰度图，再转为二值图
二值图
进行一系列腐蚀和膨胀操作
使用contours = measure.find_contours(img, 0.9)  来得到边界信息
print('共有%d个硬币'%(len(contours)))  #得到多少个边界，即多少个硬币、米粒



3代码部分：
import skimage.morphology as sm
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure,draw
from numpy import *
from skimage import io
x=io.imread('C:/Users/Administrator/Desktop/py/rice.jpg',as_grey=True)
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
numcoin=len(contours)                                 #边界个数就是米粒个数
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
title('有'+str(numcoin)+'个米粒')
plt.tight_layout()

得到如下：
https://raw.githubusercontent.com/labAxiaoming/coin2/678adb94dd6f0eaf49314ed48bd8ae6a66f5f04a/2.png

可能有误差，有些黏连，调整膨胀、腐蚀的系数

调整系数
https://raw.githubusercontent.com/labAxiaoming/coin2/678adb94dd6f0eaf49314ed48bd8ae6a66f5f04a/3.png






4、计算最大尺寸米粒

last=zeros([i,j])

from numpy import *
#contours[1][:,0]=np.around(contours[1][:,0])   #把浮点型取整

#利用高数  二元函数对平面积分的思想，把每个边界内部值变为1、2、3。。。
#第一个边界内部变1，第二个内部变2......，之后求最大尺寸可以从这里面找哪个数最多
for t in range(len(contours)):
    contours[t][:,0]=np.around(contours[t][:,0], decimals=1)
    xbig=int(max(np.around(contours[t][:,0])))    #x最大值
    xsma=int(min(np.around(contours[t][:,0])))   #x最小值
    for x in range(xsma,xbig+1):
        cc=find(np.around(contours[t][:,0])==x)
        dd=contours[t][cc,1]
        ysma=int(min(dd))                              #x对应的y的最小值，(一般有两个或者以上) 
        ybig=int(max(dd))                              #x对应的y的最大值，(一般有两个或者以上) 
        for y in range(ysma,ybig+1):
            if yyy[x,y]==1:                                  #等于1的部分变成t  （第t个边界内部）
                last[x,y]=t
            else:
                last[x,y]=0
    
io.imshow(last)

tj=zeros(int(last.max()))
for iii in range(1,int(last.max())+1):
    tj[iii-1]=len(find(np.around(last[:][:])==iii))
int(find(tj==tj.max())+1)


subplot(2,2,1)
x=io.imread('C:/Users/Administrator/Desktop/py/rice.jpg',as_grey=True)
io.imshow(x)
title('原图')

subplot(2,2,2)
gray()
io.imshow(last)
title('%d个米粒'%(last.max()))


y5=zeros([i,j])
subplot(2,2,3)
for m in range(i):                         #显示最大尺寸的米粒
    for n in range(j):
        if last[m,n]==find(tj==tj.max())+1:
            y5[m,n]=x[m,n]
        else:
            y5[m,n]=0
gray()
imshow(y5)
title('最大的一粒尺寸为%d像素'%(tj.max()))
得到下图

https://raw.githubusercontent.com/labAxiaoming/coin2/678adb94dd6f0eaf49314ed48bd8ae6a66f5f04a/5.png





