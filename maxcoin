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
for m in range(i):                         #显示最大尺寸的硬币
    for n in range(j):
        if last[m,n]==find(tj==tj.max())+1:
            y5[m,n]=x[m,n]
        else:
            y5[m,n]=0
gray()
imshow(y5)
title('最大的一个尺寸为%d像素'%(tj.max()))
