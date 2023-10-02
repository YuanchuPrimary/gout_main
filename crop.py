import sys
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
import time
# data 
import cv2
from PIL import Image
from PIL import ImageEnhance
from torchvision import transforms as tfs
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import random
import numpy as np
import torch
import random
from torchvision import models
import torch.nn as nn
from skimage.transform import resize
from scipy import ndimage
os.chdir(os.path.join(os.getcwd(),'gout_main'))
def cv_imread(file_path):
 cv_img=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
 return cv_img
def cv_imwrite(file_path,img):
   
   img_write = cv2.imencode(".bmp", img)[1].tofile(file_path)
   return img_write
def img_Sobel(img_car_gray):
    
    
    x = cv2.Sobel(img_car_gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img_car_gray, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return sobel
numm=0
for x in os.listdir('gout_data_update8_31'):
    y=os.path.join('gout_data_update8_31',x)
    for z in os.listdir(y):
      
        t=os.path.join(y,z)
        for s1 in os.listdir(t):
            m=os.path.join(t,s1)
            for p1 in os.listdir(m):
              q2=os.path.join(m,p1)
              for q3 in os.listdir(q2):
                q1=os.path.join(q2,q3)
                if'bmp' in q1:
                    img1=cv_imread(q1)
                    img=img1.copy()
                    kernelx = np.ones((6,6), np.uint8)
                    kernely = np.ones((3,3), np.uint8)
                    img = cv2.GaussianBlur(img, (3, 3), 0) 
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    #cv2.imshow('Sobel',img)
                    #cv2.waitKey(0)
                    img=cv2.threshold(img, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU )[1]
                    #cv2.imshow('Sobel',img)
                    #cv2.waitKey(0)
                    img= cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernelx)
                    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernelx)
                    img= cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernelx)
                    #cv2.imshow('Sobel',img)
                    #cv2.waitKey(0)
                    for i in range (0,1000):
                        if img[i,300]!=0 and img[i,400]!=0 and img[i,700]!=0:
                            s=i
                            break
                    #img=img1[i-20:i+380,200:800]
                    l=180
                    r=880
                    u=i+20
                    d=i+340
                    #l,r,u,d=170,880,i+20,i+400
                    plate=[[l,u],[l,d],[r,d],[r,u]]
                    plate=np.int0(plate)
                    #cv2.drawContours(img1, [plate], 0, (255, 255, 0), 1)
                    #cv2.imshow('Sobel',img1)
                    #cv2.waitKey(0)
                    img=img1[u:d,l:r]
                    pp=os.path.join('gout_new_9_9',x,z,s1,p1)
                    #pp=os.path.join('gout_crop_700320',x,z)
                    #os.remove(pp)
                    if not os.path.exists(pp):
                       os.makedirs(pp)
                    cv_imwrite(os.path.join(pp,q3),img)
                    numm+=1
                    if numm%100==1:
                      print(numm)