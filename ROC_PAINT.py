
"""
Classification cv stratified

"""
import sys
import cv2
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
import time
from arguments import args, num_classes, classes
# data 
from PIL import Image
from torchvision import transforms as tfs
import torch.nn as nn
from dataset import ImgDataset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import utilz as ut
import torch
import os
import random
from utils import GradCAM
from sklearn.metrics import roc_curve, auc
import math
import pydicom
import pandas as pd
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
start_time = time.time()
torch.backends.cudnn.benchmark = True
os.chdir(os.path.join(os.getcwd(),'gout_main'))

################### directory for saving results ###################
#from cv_classification_copy import pat_train_val, pat_label_train_val
pat_train_val, pat_label_train_val=[],[]
for m in os.listdir(args['data']):
    t=os.path.join(args['data'], m)
    for x in os.listdir(t):
        y = os.path.join(t, x)
        pat_train_val.append(y)
        pat_label_train_val.append(m)
pat_train_val, pat_label_train_val = np.array(
    pat_train_val), np.array(pat_label_train_val)
skf = StratifiedKFold(n_splits=args['cv_splits'],shuffle=True,random_state=2)
skf.get_n_splits(pat_train_val, pat_label_train_val)
id='roc_paint_2'
acc_weights=[]
sen_weights=[]
spe_weights=[]
cv_counter=0
score_all_3,score_all_34,score_all_50,score_all_v,score_all_d,score_all_18=[],[],[],[],[],[]
label_all_3,label_all_34,label_all_50,label_all_v,label_all_d,label_all_18=[],[],[],[],[],[]
fpr1=[]
tpr1=[]
roc_auc1=[]
final_val_acc = list()
final_val_spec = list()
final_val_sens = list()
position='2'
num=0
numall=0
for train_index, val_index in skf.split(pat_train_val, pat_label_train_val):
    img_train_1, img_test = pat_train_val[train_index], pat_train_val[val_index]
    img_train_label_1,img_test_label_1=pat_label_train_val[train_index],pat_label_train_val[val_index]
    num0,num1=0,0
    cv_counter+=1
    img_test_list_final = list()
    labels_test_list_final = list()
    for img_path in img_test:
            #if 'dcm' in img_path:
                #img_test_list_final.append(img_path)
                #p_label=int(img_path.split('/')[2][0:3])
        for a in os.listdir(str(img_path)):
              #if a==position:
                 t=os.path.join(img_path,a)
                 for b in os.listdir(t):
                   ee=os.path.join(t,b)
                   for c in os.listdir(ee):
                    qq=os.path.join(ee,c)
                    img_test_list_final.append(qq)
                    num+=1
                    if b=='0':
                        p_label=0
                    if b=='1':
                        p_label=1
                    #print(img_p)
                    if p_label==0:
                        labels_test_list_final.append(p_label)
                        num0+=1
                    if p_label==1:
                        
                           labels_test_list_final.append(p_label)
                           num1+=1
    print(num0,num1)
    acc_weights.append(num0+num1)
    sen_weights.append(num1)
    spe_weights.append(num0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if __name__ == "__main__":
        test_data = ImgDataset(img_test_list_final,
                              labels_test_list_final, 'test')
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False)
        
        new_dir = os.path.join(os.getcwd(), 'test_results/', id)
        new_dir_1= os.path.join(os.getcwd(), 'results/', '56_resnet_3')
        new_dir_2= os.path.join(os.getcwd(), 'results/', '45')
        new_dir_3= os.path.join(os.getcwd(), 'results/', '44')
        new_dir_4= os.path.join(os.getcwd(), 'results/', '46')
        new_dir_5= os.path.join(os.getcwd(), 'results/', '49')
        new_dir_6= os.path.join(os.getcwd(), 'results/', '47')
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        model = ut.initialize_model('resnet_3', num_classes, False, use_pretrained=True)
        model.load_state_dict(torch.load(os.path.join(os.path.join(new_dir_1,
                                                                '{}_model_classification_trained.pt'.format(cv_counter)))))
        model.eval()
        model.to(device)      
        for idx,d in enumerate(test_loader):
            data_te, target_te,filename = d['img'].to(device), d['lab'].to(device),d['file']
            outputs_te = model(data_te.float())
            [t.cpu().numpy() for t in target_te]
            label_all_3.append(int(target_te))
            c=outputs_te[0][1].cpu().detach().numpy()
            d=1/(1 + math.exp(-c))
            score_all_3.append(d)  
        print(cv_counter,'resnet_3')
        
        model = ut.initialize_model('resnet18', num_classes, False, use_pretrained=True)
        model.load_state_dict(torch.load(os.path.join(os.path.join(new_dir_6,
                                                                '{}_model_classification_trained.pt'.format(cv_counter)))))
        model.eval()
        model.to(device)      
        for idx,d in enumerate(test_loader):
            data_te, target_te,filename = d['img'].to(device), d['lab'].to(device),d['file']
            outputs_te = model(data_te.float())
            [t.cpu().numpy() for t in target_te]
            label_all_18.append(int(target_te))
            c=outputs_te[0][1].cpu().detach().numpy()
            d=1/(1 + math.exp(-c))
            score_all_18.append(d)  
        print(cv_counter,'resnet18')

        model = ut.initialize_model('resnet34', num_classes, False, use_pretrained=True)
        model.load_state_dict(torch.load(os.path.join(os.path.join(new_dir_2,
                                                                '{}_model_classification_trained.pt'.format(cv_counter)))))
        model.eval()
        model.to(device)      
        for idx,d in enumerate(test_loader):
            data_te, target_te,filename = d['img'].to(device), d['lab'].to(device),d['file']
            outputs_te = model(data_te.float())
            [t.cpu().numpy() for t in target_te]
            label_all_34.append(int(target_te))
            c=outputs_te[0][1].cpu().detach().numpy()
            d=1/(1 + math.exp(-c))
            score_all_34.append(d)  
        print(cv_counter,'resnet34')

        model = ut.initialize_model('resnet50', num_classes, False, use_pretrained=True)
        model.load_state_dict(torch.load(os.path.join(os.path.join(new_dir_3,
                                                                '{}_model_classification_trained.pt'.format(cv_counter)))))
        model.eval()
        model.to(device)      
        for idx,d in enumerate(test_loader):
            data_te, target_te,filename = d['img'].to(device), d['lab'].to(device),d['file']
            outputs_te = model(data_te.float())
            [t.cpu().numpy() for t in target_te]
            label_all_50.append(int(target_te))
            c=outputs_te[0][1].cpu().detach().numpy()
            d=1/(1 + math.exp(-c))
            score_all_50.append(d)  
        print(cv_counter,'resnet50')

        model = ut.initialize_model('vgg11', num_classes, False, use_pretrained=True)
        model.load_state_dict(torch.load(os.path.join(os.path.join(new_dir_4,
                                                                '{}_model_classification_trained.pt'.format(cv_counter)))))
        model.eval()
        model.to(device)      
        for idx,d in enumerate(test_loader):
            data_te, target_te,filename = d['img'].to(device), d['lab'].to(device),d['file']
            outputs_te = model(data_te.float())
            [t.cpu().numpy() for t in target_te]
            label_all_v.append(int(target_te))
            c=outputs_te[0][1].cpu().detach().numpy()
            d=1/(1 + math.exp(-c))
            score_all_v.append(d)  
        print(cv_counter,'vgg11')

        model = ut.initialize_model('densenet121', num_classes, False, use_pretrained=True)
        model.load_state_dict(torch.load(os.path.join(os.path.join(new_dir_5,
                                                                '{}_model_classification_trained.pt'.format(cv_counter)))))
        model.eval()
        model.to(device)      
        for idx,d in enumerate(test_loader):
            data_te, target_te,filename = d['img'].to(device), d['lab'].to(device),d['file']
            outputs_te = model(data_te.float())
            [t.cpu().numpy() for t in target_te]
            label_all_d.append(int(target_te))
            c=outputs_te[0][1].cpu().detach().numpy()
            d=1/(1 + math.exp(-c))
            score_all_d.append(d)  
        print(cv_counter,'densenet121')
            
       
       
        


plt.figure
lw = 2
fpr, tpr, thread = roc_curve(label_all_3, score_all_3)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ResNet with 3blocks ROC =%0.4f' % roc_auc)
fpr, tpr, thread = roc_curve(label_all_18, score_all_18)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='pink',
         lw=lw, label='ResNet18 ROC =%0.4f' % roc_auc)
fpr, tpr, thread = roc_curve(label_all_34, score_all_34)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='green',
         lw=lw, label='ResNet34 ROC =%0.4f' % roc_auc)
fpr, tpr, thread = roc_curve(label_all_50, score_all_50)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ResNet50 ROC =%0.4f' % roc_auc)
fpr, tpr, thread = roc_curve(label_all_v, score_all_v)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue',
         lw=lw, label='VGG11 ROC =%0.4f' % roc_auc)
fpr, tpr, thread = roc_curve(label_all_d, score_all_d)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='purple',
         lw=lw, label='DenseNet121 ROC =%0.4f' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('')
plt.legend(loc="lower right")
plt.savefig(os.path.join(new_dir, 'roc.png'))
plt.show()
plt.close()
print(num)