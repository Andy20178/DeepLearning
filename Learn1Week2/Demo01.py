# def dayup(df):
#     dayup=1
#     for i in range(365):
#         if i%7 in [6,0]:
#             dayup=dayup*(1-0.01)
#         else:
#             dayup=dayup*(1+df)
#     return dayup
# factor=0.01
# while dayup(factor)<37.78:
#     factor+=0.0001
# print(factor)
#练习搜索字符串
# weekname="星期一二三四五六日"
# weeknum=int(input("请输入数字："))
# str=weekname[0:2]+weekname[weeknum+1]
# print(str)
#打印星座标志
# for i in range(12):
#     print(chr(9800+i),end=" ")
#进度条
# import time
# scale=50
# print("开始执行".center(scale//2,"-"))
# start=time.perf_counter()
# for i in range(scale+1):
#     a="*"*i
#     b="."*(scale-i)
#     c=(i/scale)*100
#     dur=time.perf_counter()-start
#     print("\r{:^3.0f}%[{}-->{}]{:.2f}".format(c,a,b,dur),end="")
#     time.sleep(0.1)
# print("\n","执行结束".center(scale//2,"-"))
#天天向上的力量
# def daydayup(factor):
#     a=1
#     for i in range(365):
#         if i%7 in [6,0]:
#             a=a*(1-0.01)
#         else:
#             a=a*(1+factor)
#     return a
# b=pow(1.01,365)
# #print("{:.2f}".format(b))
# factor=0.01
# a=daydayup(factor)
# #print("{:.2f}".format(a))
# while a<b:
#     factor+=0.001
#     a=daydayup(factor)
# print("工作日的努力参数是: {:.3f}".format(factor))
# import time
# scale=50
# print("执行开始".center(scale//2,"-"))
# start=time.perf_counter()
# for i in range(scale+1):
#     a="*"*i
#     b="."*(scale-i)
#     c=(i/scale)*100
#     dur=time.perf_counter()-start
#     print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c,a,b,dur),end="")
#     time.sleep(0.1)
# print("\n"+"执行结束".center(scale//2,"-"))
# a=eval(input())
# b=pow(a,3)
# print("{:-^20}".format(b))
#
# N=eval(input())
# str="*"
# for i in range(1,int(N//2+1)+1):
#     str="*"*(i*2-1)
#     print(str.center(N," "))
# str=input()
# num=len(str)
# str_1=""
# for i in range(num):
#     if ord(str[i])>96 and ord(str[i])<123:
#           str_1=str_1+chr((ord(str[i])-97+3)%26+97)
#     elif ord(str[i])>64 and ord(str[i])<91:
#         str_1=str_1+chr((ord(str[i])-65+3)%26+65)
#     elif str[i]==" ":
#         str_1=str_1+" "
#     else:
#         str_1=str_1+chr(ord(str[i]))
# print(str_1)
# a=eval(input())
# b=pow(a,0.5)
# print("{:+>30.3f}".format(b))
# str=input()
# str=str.split("-")
# #print(len(str))
# print(str[0],"+",str[len(str)-1],sep="")
# try:
#     guess=eval(input())
#     print("猜{}了".format("对" if guess==99 else "错"))
# except:
#     print("输入错误")
# tall,weight=eval(input())
# BMI=weight/(pow(tall,2))
# print("BMI数值为:{:.2f}".format(BMI))
# print("BMI指标为:",end="")
# if BMI<18.5:
#     print("国际'偏瘦'",end="")
# elif BMI>18.5 and BMI<25:
#     print("国际'正常'",end="")
# elif BMI>25 and BMI<30:
#     print("国际'偏胖'", end="")
# else:
#     print("国际'肥胖'", end="")
# if BMI<18.5:
#     print(",国内'偏瘦'",end="")
# elif BMI>18.5 and BMI<24:
#     print(",国内'正常'",end="")
# elif BMI>24 and BMI<28:
#     print(",国内'偏胖'", end="")
# else:
#     print(",国内'肥胖'", end="")
# import random
# import time
# random.seed(123)
# N=eval(input())
# S=0
# start=time.perf_counter()
# for i in range(N):
#     a,b=random.random(),random.random()
#     if (pow(a,2)+pow(b,2))<=1:
#         S=S+1
# pi=S/N*4
# dur=time.perf_counter()-start
# print("{:.6f}".format(pi))
# print("{}".format(dur))
# N=100
# pi=0
# for i in range(N):
#     pi=pi+(1/pow(16,i)*(4/(8*i+1)-2/(8*i+4)-1/(8*i+5)-1/(8*i+6)))
# print("{:.6f}".format(pi))
# sum=0
# for i in range(int(966/2)):
#     sum=sum-1
# print(sum)
# gewei=0
# shiwei=0
# baiwei=0
# qianwei=0
# a=0
# for i in range(1000,10000):
#     qianwei=i//1000
#     baiwei=(i-qianwei*1000)//100
#     shiwei=(i-qianwei*1000-baiwei*100)//10
#     gewei=(i-qianwei*1000-baiwei*100-shiwei*10)
#     if (pow(gewei,4)+pow(shiwei,4)+pow(baiwei,4)+pow(qianwei,4)==i):
#         print(i)
# flag=0
# jishu=0
# while flag==0:
#     jishu = jishu + 1
#     if jishu > 3:
#         print("3次用户名或者密码均有误！退出程序。")
#         break
#     num=input()
#     key=input()
#     if num=="Kate" and key=="666666":
#         print("登录成功！")
#         break
# sum=0
# for i in range(1,100):
#     for j in range(2,i):
#         if i%j==0:
#             break
#         if j==i-1:
#             sum=sum+i
#             print(i)
# print(sum)
import numpy as np
import time
#比较numpy和手写的运行速度的区别
#手写
# a=np.random.rand(1000000)
# b=np.random.rand(1000000)
# start_1=time.time()
# c=0
# for i in range(1000000):
#     c=c+a[i]*b[i]
# dur_1=1000*(time.time()-start_1)
# print(dur_1)
# print(c)
# #用numpy算
# start_2=time.time()
# d=np.dot(a,b)
# dur_2=1000*(time.time()-start_2)
# print(dur_2)
# print(d)
import math
import numpy as np
def basic_sigmoid(x):
    """
    计算一个数的sigmoid值
    :param x:
    :return: s--sigmoid
    """
    s=1/(1+np.exp(-x))
    return s
def sigmoid_derivative(x):
    """
    计算sigmoid的梯度，der=s(1-s)
    :param x:
    :return:
    """
    s=basic_sigmoid(x)
    ds=s*(1-s)
    return ds
# a=np.array([1,2,3])
# s=basic_sigmoid(a)
# ds=sigmoid_derivative(a)
# #使用numpy实现sigmoid函数
# print(s)
# print(ds)
def image2vector(image):
    """
    输入一张图片的向量(length,height,depth)
    :param image:
    :return: v--a vector of shape(length*height*depth,1)
    """
    v=image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
    return v
# image = np.array([[[ 0.67826139,  0.29380381],
#         [ 0.90714982,  0.52835647],
#         [ 0.4215251 ,  0.45017551]],
#
#        [[ 0.92814219,  0.96677647],
#         [ 0.85304703,  0.52351845],
#         [ 0.19981397,  0.27417313]],
#
#        [[ 0.60659855,  0.00533165],
#         [ 0.10820313,  0.49978937],
#         [ 0.34144279,  0.94630077]]])
# a=image2vector(image)
# print (a)
def normalizeRows(x):
    """
    对输入的矩阵进行归一化
    :param x:
    :return:
    """
    x_norm=np.linalg.norm(x,axis=1,keepdims=True)
    x_normalized=x/x_norm
    return x_normalized
# x = np.array([
#     [0, 3, 4],
#     [1, 6, 4]])
# print(normalizeRows(x))
#
# x_norm=normalizeRows(x)
# x_normliazed=x/x_norm
# print(x_norm)
# print("正确的输出:",end="")
# print(x_norm.shape,x_normliazed)
# x_norm_2=x.sum(axis=1)
# print(x_norm_2.shape)
# x_norm_2=x_norm_2.reshape(2,1)
# print("reshape后的结果：",end="")
# print(x_norm_2.shape)
# print(x_norm_2)
# x_normliazed_2=x/x_norm_2
# print(x_normliazed_2)
def softmax(x):
    """
    为每一个输入的矩阵计算其softmax的值
    :param x:
    :return:
    """
    x_exp=np.exp(x)
    x_sum=np.sum(x_exp,axis=1,keepdims=True)
    x_softmax=x_exp/x_sum
    return x_softmax
"""
比较向量化与非向量化的区别
"""
# x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
# x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
# ###非向量化的点乘，即x1Tx2
# sum=0
# tic=time.perf_counter()
# for i in range(len(x1)):
#     sum+=x1[i]*x2[i]
# toc=time.perf_counter()
# print("dot="+str(sum)+"\n-------computeation time ="+str(1000*(toc-tic)))
# ###向量化方法实现
# sum=0
# tic=time.perf_counter()
# sum=np.dot(x1,x2)
# toc=time.perf_counter()
# print("dot="+str(sum)+"\n-------computeation time ="+str(1000*(toc-tic)))
#
# ###非向量化的外积
# outer = np.zeros((len(x1), len(x2)))#python的二维数组要用双括号括起来
# for i in range(len(x1)):
#     for j in range(len(x2)):
#         outer[i,j]=x1[i]*x2[j]
# print("outer="+str(outer)+"\n --------------")
# ##使用向量化的方法做外积
# outer=np.outer(x1,x2)
# print(str(outer))
#
# ###传统方法实现逐个元素相乘
# mul=np.zeros(len(x1))
# for i in range(len(x1)):
#     mul[i]=x1[i]*x2[i]
# print(str(mul)+"-----------")
#
# #用numpy实现逐个元素相乘
# mul=np.multiply(x1,x2)
# print(str(mul))
#
# ###传统方法实现点积
# W=np.random.rand(3,len(x1))
# gdot=np.zeros(W.shape[0])
# for i in range(W.shape[0]):
#     for j in range(len(x1)):
#         gdot+=W[i,j]*x1[j]
# print(str(W))
#
# ###使用向量化的方法
# gdot=np.dot(W,x1)

"""
L1 and L2's loss function
"""
def L1(yhat,y):
    """
    yhat---vector of size m(predicted labels)
    y--vector of size m(ture labels)
    :param yhat:
    :param y:
    :return:loss--the value of the L1 loss function defined above
    """
    loss=np.sum(np.abs(y-yhat))
    return loss
def L2(yhat,y):
    """

    yhat---vector of size m(predicted labels)
    y--vector of size m(ture labels)
    :param yhat:
    :param y:
    :return:loss--the value of the L2 loss function defined above
    """
    loss=np.dot((y-yhat),(y-yhat).T)
    return loss
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))
print("L1 = " + str(L1(yhat,y)))