
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
new_dir = os.path.join(os.getcwd(), 'results/', args['id'])
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
    print('Directory ', new_dir, 'created!')
else:
    print('Directory ', new_dir, 'already exists!')

file_paths = list()
labels_list_initial = list()
img_val = list()
pat_train_val=list()
pat_test=list()
pat_label_train_val=list()
pat_label_test=list()
#aug_list=['(1)','(1)rotate0','(1)rotate1','(2)',
 #        '(2)rotate0','(2)rotate1','crop','croprotate0','croprotate1',
 #             'rotate0','rotate1']
 #        '(2)rotate0','(2)rotate1','crop','croprotate0','croprotate1',
#aug_list=['(1)','(2)','(2)rotate0']
#for m in os.listdir(args['data']):
  #if m=='train' or m=='test':
    #t=os.path.join(args['data'], m)      
    #for x in os.listdir(t) : 
            #y = os.path.join(t, x)
            #for a in os.listdir(y):
             # z = os.path.join(y, a)
              #if m=='train':
               #   pat_train_val.append(z)
               #   pat_label_train_val.append(int(x[:1]))
              #if m=='test1':
                #  pat_test.append(z)
                 # pat_label_test.append(int(x[:1]))
s=0
num0=0
num1=0
for m in os.listdir(args['data']):
    t=os.path.join(args['data'], m)
    for x in os.listdir(t):
        y = os.path.join(t, x)
        pat_train_val.append(y)
        pat_label_train_val.append(m)
        num0+=1
        #for a in os.listdir(y):
             #e= os.path.join(y, a)
             #for z in os.listdir(e):
                    #f=os.path.join(e,z)
                  #ds=pydicom.dcmread(e)
                  #img= ds.pixel_array
                  #if len(img.shape)!=4:
                   # s+=1
                    #if s%5==1 or s%5==2:

                    #pat_train_val.append(f)
                      #if 'GOUT' in m:
                        #pat_label_train_val.append(1)
                        #num1+=1
                      #if 'OB' in m:
                       # pat_label_train_val.append(0)
                       # num0+=1
print(num0,num1)



# split list into train and test
# split train into train and val cross validated
#pat_train_val, pat_test, pat_label_train_val, pat_label_test = train_test_split(
#    file_paths, labels_list_initial, test_size=args['test_split'], train_size=args['train_split'], random_state=args['seed'], stratify=labels_list_initial)


# =============================================================================
# cross validation
# =============================================================================
skf = StratifiedKFold(n_splits=args['cv_splits'],shuffle=True,random_state=2)
skf.get_n_splits(pat_train_val, pat_label_train_val)
final_val_acc = list()
final_val_spec = list()
final_val_sens = list()
final_test_true1=list()
final_test_true2=list()
final_test_true3=list()
train_acc1=[]
val_acc1=[]
train_loss1=[]
val_loss1=[]
fpr1=[]
tpr1=[]
roc_auc1=[]
score_all=[]
label_all=[]
acc_weights=[]
sen_weights=[]
spe_weights=[]
train_score=[]
train_label=[]
val_score=[]
val_label=[]
cv_counter = 1
pat_train_val, pat_label_train_val = np.array(
    pat_train_val), np.array(pat_label_train_val)
pat_test ,pat_label_test= np.array(pat_test),np.array(pat_label_test)
position='2'
#index=[i for i in range(len(pat_train_val))]
#random.shuffle(index)
#pat_train_val=pat_train_val[index]
#pat_label_train_val=pat_label_train_val[index]
#index=[i for i in range(len(pat_test))]
#random.shuffle(index)
#pat_test=pat_test[index]
#pat_label_test=pat_label_test[index]
for train_index, val_index in skf.split(pat_train_val, pat_label_train_val):
    img_train_1, img_test = pat_train_val[train_index], pat_train_val[val_index]
    img_train_label_1,img_test_label_1=pat_label_train_val[train_index],pat_label_train_val[val_index]
    img_train, img_val, img_train_label, img_val_label = train_test_split(
           img_train_1, img_train_label_1, test_size=args['test_split'], train_size=args['train_split'], random_state=args['seed'], stratify=img_train_label_1)
    labels_train_list_final = list()
    labels_val_list_final = list()
    labels_test_list_final = list()
    img_train_list_final = list()
    img_val_list_final = list()
    img_test_list_final = list()
    sss=0
# train
    num0=0
    num1=0
    
    k=0
    for img_path in img_train:
            #if 'dcm' in img_path:
            for a in os.listdir(str(img_path)):
              #if a==position:
                 t=os.path.join(img_path,a)
                 for b in os.listdir(t):
                   ee=os.path.join(t,b)
                   for c in os.listdir(ee):
                    qq=os.path.join(ee,c)
                    img_train_list_final.append(qq)
                    if b=='0':
                        p_label=0
                    if b=='1':
                        p_label=1
                    #print(img_p)
                    if p_label==0:
                        labels_train_list_final.append(p_label)
                        num0+=1
                        #if num0>300:
                            #break
                        k+=1
                    if p_label==1:
                           labels_train_list_final.append(p_label)
                           num1+=1
            
                    if p_label==0:
                     #if k%4==1:
                      ss=os.path.join('gout_data_augleft11_15',img_path.split('/')[1],img_path.split('/')[2],a,b,c) 
                      img_train_list_final.append(ss)
                      labels_train_list_final.append(p_label)
                      num0+=1 
                     #if k%4==3:
                      ss=os.path.join('gout_data_augup6_10',img_path.split('/')[1],img_path.split('/')[2],a,b,c)
                      img_train_list_final.append(ss)
                      labels_train_list_final.append(p_label)
                      num0+=1 
                    else:  
                      ss=os.path.join('gout_data_augleft11_15',img_path.split('/')[1],img_path.split('/')[2],a,b,c) 
                      img_train_list_final.append(ss)
                      labels_train_list_final.append(p_label)
                      num1+=1 
                      ss=os.path.join('gout_data_augup6_10',img_path.split('/')[1],img_path.split('/')[2],a,b,c)
                      img_train_list_final.append(ss)
                      labels_train_list_final.append(p_label)
                      num1+=1 
    print(num0,num1)
    if num0<num1:
        aug_x='0'
    else:
        aug_x='1'
    #aug_x='1122'
    for m in os.listdir('gout_new_9_9'):
     t=os.path.join('gout_new_9_9', m)
     for x in os.listdir(t):
        y = os.path.join(t, x)
        for a in os.listdir(str(img_path)):
              #if a==position:
                 r=os.path.join(img_path,a)
                 for b in os.listdir(r):
                   ee=os.path.join(r,b)
                   for c in os.listdir(ee):
                    qq=os.path.join(ee,c)
                    img_train_list_final.append(qq)
                    if b=='0':
                        p_label=0
                    if b=='1':
                        p_label=1
                    #print(img_p)
                    
                    labels_train_list_final.append(p_label)
    for img_path in img_train:
            
            #if 'dcm' in img_path:
            for a in os.listdir(str(img_path)):
              #if a==position:
                 t=os.path.join(img_path,a)
                 for b in os.listdir(t):
                   ee=os.path.join(t,b)
                   for c in os.listdir(ee):
                    if num0==num1:
                        break
                    qq=os.path.join(ee,c)
                    if b==aug_x:
                      ss=os.path.join('gout_data_augright6_10',img_path.split('/')[1],img_path.split('/')[2],a,b,c)
                      img_train_list_final.append(ss)
                      p_label=int(aug_x)
                    #print(img_p)
                      if p_label==0:
                        labels_train_list_final.append(p_label)
                        num0+=1  
                      if p_label==1:
                        
                           labels_train_list_final.append(p_label)
                           num1+=1  
    print(num0,num1)
    if num0<num1:
        aug_x='0'
    else:
        aug_x='1'
    #aug_x='1122'
    for img_path in img_train:
            
            #if 'dcm' in img_path:
            for a in os.listdir(str(img_path)):
             # if a==position:
                 t=os.path.join(img_path,a)
                 for b in os.listdir(t):
                   ee=os.path.join(t,b)
                   for c in os.listdir(ee):
                    if aug_x=='0':
                      if num0>=num1:
                        break
                    else:
                        if num1>=num0:
                            break
                    qq=os.path.join(ee,c)
                    if b==aug_x:
                      ss=os.path.join('gout_data_augleft6_10',img_path.split('/')[1],img_path.split('/')[2],a,b,c)
                      img_train_list_final.append(ss)
                      p_label=int(aug_x)
                      if p_label==0:
                        labels_train_list_final.append(p_label)
                        num0+=1  
                      if p_label==1:
                           labels_train_list_final.append(p_label)
                           num1+=1
                      ss=os.path.join('gout_data_augright11_15',img_path.split('/')[1],img_path.split('/')[2],a,b,c)
                      img_train_list_final.append(ss)
                      p_label=int(aug_x)
                      if p_label==0:
                        labels_train_list_final.append(p_label)
                        num0+=1  
                      if p_label==1:
                           labels_train_list_final.append(p_label)
                           num1+=1 
                      ss=os.path.join('gout_data_augdown6_10',img_path.split('/')[1],img_path.split('/')[2],a,b,c)
                      img_train_list_final.append(ss)
                      p_label=int(aug_x)
                      if p_label==0:
                        labels_train_list_final.append(p_label)
                        num0+=1  
                      if p_label==1:
                           labels_train_list_final.append(p_label)
                           num1+=1     
                    
    print(num0,num1)  
    img_train_list_final,labels_train_list_final=np.array(img_train_list_final),np.array(labels_train_list_final)
    index=[i for i in range(len(img_train_list_final))]
    random.shuffle(index)
    img_train_list_final=img_train_list_final[index]
    labels_train_list_final=labels_train_list_final[index]
    img_train_list_final,labels_train_list_final=list(img_train_list_final),list(labels_train_list_final)
    
# val
    num0,num1=0,0
    for img_path in img_val:
            #if 'dcm' in img_path:
                
            for a in os.listdir(str(img_path)):
              #if a==position:
                 t=os.path.join(img_path,a)
                 for b in os.listdir(t):
                   ee=os.path.join(t,b)
                   for c in os.listdir(ee):
                    qq=os.path.join(ee,c)
                    img_val_list_final.append(qq)
                    if b=='0':
                        p_label=0
                    if b=='1':
                        p_label=1
                    #print(img_p)
                    if p_label==0:
                        labels_val_list_final.append(p_label)
                        num0+=1
                    if p_label==1:
                        
                           labels_val_list_final.append(p_label)
                           num1+=1
    print(num0,num1)         

# test
    num0,num1=0,0
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


    if __name__ == "__main__":

        if cv_counter == 1:
            print('number of files for training: ', len(img_train_list_final))
            print('number of files for validation: ', len(img_val_list_final))
            print('number of files for testing: ', len(img_test_list_final))

        # custom dataloader
       # img=Image.open(img_train_list_final)
       #tfs=tfs.Compose([tfs.Resize(120),tfs.RandomHorizontalFlip(),tfs.RandomCrop(96),tfs.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5)])
        train_data = ImgDataset(img_train_list_final,
                                labels_train_list_final, 'train')
        val_data = ImgDataset(img_val_list_final,
                              labels_val_list_final, 'val')
        test_data = ImgDataset(img_test_list_final,
                              labels_test_list_final, 'test')                     
        
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args['batch_size'], shuffle=False, num_workers=6,pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=args['batch_size'], shuffle=False, num_workers=6,pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False)
        # =============================================================================
        # architecture
        # =============================================================================

        # Initialize the model
        model = ut.initialize_model(
            args['model_name'], num_classes, args['feature_extract'], use_pretrained=True)

        params_to_update = model.parameters()
        #print("Params to learn:")
        if args['feature_extract']:
            params_to_update = []
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    # print("\t",name)
        #else:
            #for name, param in model.named_parameters():
                #if param.requires_grad == True:
                    #print("\t", name)

        # =============================================================================
        # median frequency balancing
        # =============================================================================
        # get the class labels of each image
        class_labels = labels_train_list_final
        # empty array for counting instance of each class
        count_labels = np.zeros(len(classes))
        # empty array for weights of each class
        class_weights = np.zeros(len(classes))

        # populate the count array
        for l in class_labels:
            count_labels[l] += 1
            #count_labels[l] =10

        # get median count
        median_freq = np.median(count_labels)

        classes.sort()
        # calculate the weigths
       # class_weights[0]=1
        #class_weights[1]=2
        for i in range(len(classes)):
            print(i)
            print(class_weights[i])
            class_weights[i] = median_freq/count_labels[i]
            # print the weights
            #print('weights: ', classes[i], ":", class_weights[i])
        
            print('weights: ', classes[i], ":", class_weights[i])
        # =============================================================================
        #  optimizer, loss function
        # =============================================================================
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(class_weights)
        optimizer = optim.Adam(params_to_update, lr=args['lr'])
        # model = torch.nn.DataParallel(model)
        model.to(device)

        ################### training ###################
        valid_loss_min = 10000
        val_loss = []
        val_acc = []
        train_loss = []
        train_acc = []
        test_loss = []
        test_true1,test_true2,test_true3=[],[],[]
        test_acc = []
        val_sens, val_spec = [], []
        val_true1,val_true2,val_true3=[],[],[]

        
        test_sens, test_spec = [], []
        total_step_train = len(train_loader)
        total_step_val = len(val_loader)
        total_step_test=len(test_loader)
        lab_list, pred_list = list(), list()


        for epoch in range(1, args['epx']+1):
            running_loss = 0.0
            correct_t = 0
            total_t = 0
            lab_list, pred_list = list(), list()
            score_train_tmp=[]
            label_train_tmp=[]
            sensitivity_train, specificity_val_train, sensitivity_val, specificity_val = 0, 0, 0, 0
            #print('----------------------------------')
            if epoch%5==0:
                print(f'Epoch {epoch}')

            for batch_idx, d in enumerate(train_loader):
                data_t, target_t = d['img'].to(device), d['lab'].to(device)
                ### zero the parameter gradients
                optimizer.zero_grad()
                ### forward + backward + optimize
                outputs_t = model(data_t.float())

                loss_t = criterion(outputs_t, target_t.long())

                loss_t.backward()
                optimizer.step()
                ### print statistics
                running_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)
                [t.cpu().numpy() for t in target_t]
                [p.cpu().numpy() for p in pred_t]
                #pred_t=pred_t.cpu().numpy().tolist()
                #target_t=target_t.cpu().numpy().tolist()
                pred_list.append(pred_t)
                lab_list.append(target_t)
                out_list=outputs_t.cpu().detach().numpy()
                for i in out_list:
                   c=i[1]
                   d=1/(1 + math.exp(-c))
                   score_train_tmp.append(d)
                #lab_list=lab_list.cpu().detach().numpy()
            for i in lab_list:
                for j in i:
                    label_train_tmp.append(j.cpu().detach().numpy())
                ### visualize training images
                #if batch_idx < 20:
                #     img = data_t.cpu().numpy()
                 #    plt.figure(dpi=300)
                 #    plt.axis('off')
                 #    plt.imshow(img[0,0,:,:], cmap='bone')
                     
            senstt, spectt, _ = ut.calculate_sensitivity_specificity(
                    lab_list, pred_list, num_classes)

            train_acc.append(100 * (correct_t / total_t))
            train_loss.append(running_loss / total_step_train)
            #print(
            #    f'\ntrain loss: {(train_loss[-1]):.4f}, train acc: {(train_acc[-1]):.4f}')

            ################ validation ###################
            batch_loss = 0
            total_v = 0
            correct_v = 0
            lab_list,pred_list=[],[]
            val_score_tmp=[]
            label_val_tmp=[]
            with torch.no_grad():
                model.eval()
                for d in (test_loader):
                    data_v, target_v = d['img'].to(device), d['lab'].to(device)
                    outputs_v = model(data_v.float())

                    [t.cpu().numpy() for t in target_v]
                    lab_list.append(target_v)

                    loss_v = criterion(outputs_v, target_v.long())
                    batch_loss += loss_v.item()
                    pred, pred_v = torch.max(outputs_v, dim=1)
                    pred = pred.cpu()  # .numpy()

                    [p.cpu().numpy() for p in pred_v]
                    pred_list.append(pred_v)

                    correct_v += torch.sum(pred_v == target_v).item()
                    total_v += target_v.size(0)
                    out_list=outputs_v.cpu().detach().numpy()
                    for i in out_list:
                       c=i[1]
                       d=1/(1 + math.exp(-c))
                       val_score_tmp.append(d)
                    #lab_list=lab_list.cpu().detach().numpy()
                for i in lab_list:
                      for j in i:
                        label_val_tmp.append(j.cpu().detach().numpy())
                    
                    ### visualize validation images
                    #img = data_v.cpu().numpy()
                    #plt.figure()
                    #plt.axis('off')
                    #plt.imshow(img[0,0,:,:])
                ### specificity = tn/(tn + fp)
                if num_classes<=2:
                 sens, spec, _ = ut.calculate_sensitivity_specificity(
                    lab_list, pred_list, num_classes)
                else:
                    true1,true2,true3,_=ut.calculate_sensitivity_specificity(
                       lab_list, pred_list, num_classes)



                if num_classes<=2:
                    
                    val_spec.append(spec)
                    val_sens.append(sens)
                else:
                    val_true1.append(true1)
                    val_true2.append(true2)
                    val_true3.append(true3)
                val_acc.append(100*correct_v/total_v)
                val_loss.append(batch_loss / total_step_val)

                network_learned = (batch_loss < valid_loss_min ) and (epoch>10)
                #network_learned = 1< 2
                #print('validation loss: ', round(val_loss[-1], 3))
                #print('validation acc: ', round(val_acc[-1], 3))
                #print('validation spec: ', round(val_spec[-1], 3))
                #print('validation sens: ', round(val_sens[-1], 3))
                # Saving the best weight
                if network_learned:
                    print('val:epoch',epoch)
                    valacc=100*correct_v/total_v
                    valsens=sens
                    valspec=spec
                    trainacc=train_acc[-1]
                    trainsen=senstt
                    trainspe=spectt
                    train_score_list=score_train_tmp
                    train_label_all=label_train_tmp
                    val_score_list=val_score_tmp
                    val_label_all=label_val_tmp
                    valid_loss_min = batch_loss
                    torch.save(model.state_dict(), os.path.join(new_dir,
                                                                '{}_model_classification_trained.pt'.format(cv_counter)))
                    model.eval()
                    lab_list_test,pred_list=list(),list()
                    batch_loss = 0
                    total_test = 0
                    correct_test = 0
                    for d in (test_loader):
                        data_te, target_te = d['img'].to(device), d['lab'].to(device)
                        outputs_te = model(data_te.float())

                        [t.cpu().numpy() for t in target_te]
                        lab_list_test.append(target_te)

                        loss_te = criterion(outputs_te, target_te.long())
                        batch_loss += loss_te.item()
                        pred, pred_te = torch.max(outputs_te, dim=1)
                        pred = pred.cpu()  # .numpy()

                        [p.cpu().numpy() for p in pred_te]
                        pred_list.append(pred_te)

                        correct_test += torch.sum(pred_te == target_te).item()
                        total_test += target_te.size(0)
                    if num_classes<=2:
                         sens, spec, _ = ut.calculate_sensitivity_specificity(
                              lab_list_test, pred_list, num_classes)
                    else:
                          true1,true2,true3,_=ut.calculate_sensitivity_specificity(
                                lab_list_test, pred_list, num_classes)
                    test_acc.append(100*correct_test/total_test)
                    if num_classes<=2:
                    
                         test_spec.append(spec)
                         test_sens.append(sens)
                    else:
                       test_true1.append(true1)
                       test_true2.append(true2)
                       test_true3.append(true3)
                    test_loss.append(batch_loss / total_step_test)
            model.train()
  
        torch.cuda.empty_cache()

        print('best validation loss: ', round(min(val_loss), 3))
        print('best validation acc: ', round(max(val_acc), 3))
        if num_classes<=2:
           print('best validation spec: ', round(max(val_spec), 3))
           print('best validation sens: ', round(max(val_sens), 3))
        else:
            print('best true1:',round(max(val_true1),3))
            print('best true2:',round(max(val_true2),3))
            print('best true3:',round(max(val_true3),3))
        print('val_acc:',valacc)
        print('val_sens:',valsens)
        print('val_spec:',valspec)
        print('trainacc:',trainacc)
        print('trainsen:',trainsen)
        print('trainspe:',trainspe)
        #print(valacc,valsens,valspec)
        #print('best validation loss: ', round(val_loss[-1], 3))
        #print('best validation acc: ', round(val_acc[-1], 3))
        #print('best validation spec: ', round(val_spec[-1], 3))
        #print('best validation sens: ', round(val_sens[-1], 3))
        final_dir = os.path.join(new_dir, str(cv_counter))
        final_dir_right= os.path.join(new_dir, str(cv_counter)+'_'+'right')
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
            print('Directory for result: ', final_dir, 'created!')
        else:
            print('Directory for result: ', final_dir, 'already exists!')
        if not os.path.exists(final_dir_right):
            os.makedirs(final_dir_right)
        model.load_state_dict(torch.load(os.path.join(os.path.join(new_dir,
                                                                '{}_model_classification_trained.pt'.format(cv_counter)))))
        model.eval()
        model.to(device)       
        target_layers = [model.layer4[-1]]
        gradcam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        true_label=[]
        pre_label=[]
        file_list=[]
        gradcam_accum=list()
        img_accum=list()
        score=[]
        for idx,d in enumerate(test_loader):
            data_te, target_te,filename = d['img'].to(device), d['lab'].to(device),d['file']
            outputs_te = model(data_te.float())

            [t.cpu().numpy() for t in target_te]
            
            pred, pred_te = torch.max(outputs_te, dim=1)
            label_all.append(int(target_te))
            c=outputs_te[0][1].cpu().detach().numpy()
            d=1/(1 + math.exp(-c))
            score.append(d)
            score_all.append(d)
            pred = pred.cpu()  # .numpy()

            [p.cpu().numpy() for p in pred_te]
            l=str(target_te.item())
            p=str(pred_te.item())
            gradcam_img = gradcam(input_tensor=data_te, target_category=target_te.item())
            gradcam_accum.append(gradcam_img[0,:,:])
            img=data_te.cpu().numpy()
            img_accum.append(img[0,0,:,:])
            gradcam_accum[idx] += gradcam_img[0,:,:]
            true_label.append(l)
            pre_label.append(p)
            file_list.append(filename)
        model.train()
        num = 0
        for img_normal, img_gradcam_accum,l,p ,qqq in zip(img_accum, gradcam_accum,true_label,pre_label,file_list):
          plt.figure() 
          plt.subplot(2, 1, 1)
          plt.title(l + '_' + p)
          plt.axis('off')
          img_org = cv2.imread(qqq[0])
          plt.imshow(img_org)
          plt.subplot(2, 1, 2)
          plt.axis('off')
          #qqq=qqq[0][26:-5]
          qqq=qqq[0].split('/')
          qqq=qqq[1]+'_'+qqq[2]+'_'+qqq[3]+'_'+qqq[5][:-5]

          #plt.title(l + '_' + p)
          plt.imshow(img_normal, cmap='bone')
          plt.imshow(img_gradcam_accum, alpha=0.5, cmap='rainbow')
          if l!=p:
             plt.savefig(os.path.join(final_dir, str(l)+'_'+str(p)+'_'+qqq), dpi=500)
          if l==p:
             plt.savefig(os.path.join(final_dir_right, str(l)+'_'+str(p)+'_'+qqq), dpi=500)
    #plt.savefig(os.path.join(final_dir, 'gradcam_image_accum' + str(num)), dpi=500)
          plt.close()
          num += 1
        plt.figure
        fpr, tpr, thread = roc_curve(labels_test_list_final, score)
        roc_auc = auc(fpr, tpr)
        fpr1.append(fpr)
        tpr1.append(tpr)
        roc_auc1.append(roc_auc)
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area=%0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(final_dir, 'roc.png'))
        plt.show()
        plt.close()
        ################### loss graphs ###################
        fig = plt.figure(figsize=(20, 10))
        plt.title("Loss")
        plt.plot(train_loss, label='train', color='c')
        train_loss1.append(train_loss)
        plt.plot(val_loss, label='validation', color='m')
        val_loss1.append(val_loss)
        plt.xlabel('num_epochs', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.legend(loc='best')
        plt.savefig(os.path.join(final_dir, 'loss'))
        plt.close()

        ################### accuracy graphs ###################
        fig = plt.figure(figsize=(20, 10))
        plt.title(" Accuracy")
        plt.plot(train_acc, label='train', linestyle='dotted', color='c')
        train_acc1.append(train_acc)
        plt.plot(val_acc, label='validation', linestyle='dotted', color='m')
        val_acc1.append(val_acc)
        plt.xlabel('num_epochs', fontsize=12)
        plt.ylabel('accuracy', fontsize=12)
        plt.legend(loc='best')
        plt.savefig(os.path.join(final_dir, 'accuracy'))
        plt.close()


        
        print('------------------------ cv split done ------------------------')

    # =============================================================================
    # saving data for final cross validated evaluation
    # =============================================================================
        #final_val_acc.append(max(val_acc))
        #final_val_sens.append(max(val_sens))
        #final_val_spec.append(max(val_spec))
        final_val_acc.append(test_acc[-1])
        if num_classes<=2:
            final_val_sens.append(test_sens[-1])
            final_val_spec.append(test_spec[-1])
        else:
            final_test_true1.append(test_true1[-1])
            final_test_true2.append(test_true2[-1])
            final_test_true3.append(test_true3[-1])
        print('cv_counter: ', str(cv_counter) + '/' + str(args['cv_splits']))
        cv_counter += 1

        print()
        print()
        print(final_dir)
        print('------------------------ FINAL RESULTS ------------------------')
        print('cross validated val accuracy: ', round(np.average(final_val_acc,weights=acc_weights), 3))
        if num_classes<=2:
            print('cross validated val sensitivity: ', round(np.average(final_val_sens,weights=sen_weights), 3))
            print('cross validated val specificity: ', round(np.average(final_val_spec,weights=spe_weights), 3))
        else:
            print('cross validated test true1: ', round(np.mean(final_test_true1), 3))
            print('cross validated test true2: ', round(np.mean(final_test_true2), 3))
            print('cross validated test true3: ', round(np.mean(final_test_true3), 3))
        print('---------------------------------------------------------------')

        end = time.time()
        elapsed_time = end - start_time
        print('elapsed mins: ', elapsed_time/60)
        print('elapsed hours: ', elapsed_time/3600)
        torch.cuda.empty_cache()
        
    
            
        

    if cv_counter > args['cv_splits']:
        with open(os.path.join(new_dir, 'args.txt'), 'w') as file:
            file.write(json.dumps(args))
x=np.average(final_val_acc,weights=acc_weights)
final_val_acc.append(x)
x=np.average(final_val_sens,weights=sen_weights)
final_val_sens.append(x)
x=np.average(final_val_spec,weights=spe_weights)
final_val_spec.append(x)
acc=np.matrix(final_val_acc)
sen=np.matrix(final_val_sens)
spe=np.matrix(final_val_spec)
acc=acc.T
sen=sen.T
spe=spe.T
ss=np.column_stack((acc,sen,spe))
ss=pd.DataFrame(ss)
ss.columns = ['Acc', 'Sen','Spe']
ss.to_excel(os.path.join(new_dir,'result.xlsx'))
fig = plt.figure(figsize=(40, 20))
plt.title("Loss")
plt.plot(train_loss1[0], label='train_1', color='red')
plt.plot(val_loss1[0], label='validation_1', color='orange')
plt.plot(train_loss1[1], label='train_2', color='yellow')
plt.plot(val_loss1[1], label='validation_2', color='green')
plt.plot(train_loss1[2], label='train_3', color='cyan')
plt.plot(val_loss1[2], label='validation_3', color='blue')
plt.plot(train_loss1[3], label='train_4', color='purple')
plt.plot(val_loss1[3], label='validation_4', color='pink')
plt.plot(train_loss1[4], label='train_5', color='magenta')
plt.plot(val_loss1[4], label='validation_5', color='brown')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.legend(loc='best')
plt.savefig(os.path.join(new_dir, 'loss'))
plt.close()
fig = plt.figure(figsize=(40, 20))
plt.title("Accuracy")
plt.plot(train_acc1[0], label='train_1', color='red')
plt.plot(val_acc1[0], label='validation_1', color='orange')
plt.plot(train_acc1[1], label='train_2', color='yellow')
plt.plot(val_acc1[1], label='validation_2', color='green')
plt.plot(train_acc1[2], label='train_3', color='cyan')
plt.plot(val_acc1[2], label='validation_3', color='blue')
plt.plot(train_acc1[3], label='train_4', color='purple')
plt.plot(val_acc1[3], label='validation_4', color='pink')
plt.plot(train_acc1[4], label='train_5', color='magenta')
plt.plot(val_acc1[4], label='validation_5', color='brown')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.legend(loc='best')
plt.savefig(os.path.join(new_dir, 'accuracy'))
plt.close()

plt.figure
fpr, tpr, thread = roc_curve(label_all, score_all)
roc_auc = auc(fpr, tpr)
#fpr1.append(fpr)
#tpr1.append(tpr)
#roc_auc1.append(roc_auc)
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area=%0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig(os.path.join(new_dir, 'roc.png'))
plt.show()
plt.close()
plt.figure
fpr, tpr, thread = roc_curve(train_label_all, train_score_list)
roc_auc = auc(fpr, tpr)
#fpr1.append(fpr)
#tpr1.append(tpr)
#roc_auc1.append(roc_auc)
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area=%0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig(os.path.join(new_dir, 'roc_train.png'))
plt.show()
plt.close()

plt.figure
fpr, tpr, thread = roc_curve(val_label_all, val_score_list)
roc_auc = auc(fpr, tpr)
#fpr1.append(fpr)
#tpr1.append(tpr)
#roc_auc1.append(roc_auc)
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area=%0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig(os.path.join(new_dir, 'val_train.png'))
plt.show()
plt.close()
#plt.figure
      
     
#lw = 2
#plt.plot(fpr1[0], tpr1[0], color='darkorange',
 #        lw=lw, label='ROC curve (area=%0.2f)' % roc_auc1[0])
#plt.plot(fpr1[1], tpr1[1], color='green',
 #        lw=lw, label='ROC curve (area=%0.2f)' % roc_auc1[1])
#plt.plot(fpr1[2], tpr1[2], color='blue',
 #        lw=lw, label='ROC curve (area=%0.2f)' % roc_auc1[2])
#plt.plot(fpr1[3], tpr1[3], color='red',
  #       lw=lw, label='ROC curve (area=%0.2f)' % roc_auc1[3])
#plt.plot(fpr1[4], tpr1[4], color='purple',
 #        lw=lw, label='ROC curve (area=%0.2f)' % roc_auc1[4])
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.savefig(os.path.join(new_dir, 'roc.png'))
#plt.show()
#plt.close()
print('acc_list',final_val_acc)
if num_classes>2:
    print('true1_list',final_test_true1)
    print('true2_list',final_test_true2)
    print('true3_list',final_test_true3)
else:
    print('sen_list',final_val_sens)
    print('spe_list',final_val_spec)




