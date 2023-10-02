"""
utility functions
"""

import numpy as np
import torch
import random
from torchvision import models
import torch.nn as nn
from skimage.transform import resize
from arguments import args
from scipy import ndimage

#########################################################################



      
def calculate_sensitivity_specificity(y_test, y_pred_test, num_classes):
    true_pos, false_pos, true_neg, false_neg, acc,true_2,num0,num1,num2 = 0,0,0,0,0,0,0,0,0
    for (l,p) in zip(y_test, y_pred_test): 
        
        for (l_, p_) in zip(l, p):
            if l_ == p_ == 1:
                true_pos += 1
            if l_ == 0 and p_ == 1:
                false_pos += 1
            if l_ == p_ == 0:
                true_neg += 1
            if l_ == 1 and p_ == 0:
                false_neg += 1
            if l_ == p_:
                acc += 1
            if num_classes>2:
                if l_==p_==2:
                    true_2+=1
                if l_==0:
                    num0+=1
                if l_==1:
                    num1+=1
                if l_==2:
                    num2+=1
            
    if num_classes > 2:
        accuracy = acc / len(y_pred_test)
        true1=true_neg/num0
        true2=true_pos/num1
        true3=true_2/num2
        return true1, true2,true3, accuracy
        
    else:
        # Calculate accuracy
        accuracy = acc / (len(y_pred_test) * args['batch_size'])
        # Calculate sensitivity and specificity
        if (true_pos + false_neg)!=0:
           sensitivity = true_pos / (true_pos + false_neg)
        else:
            sensitivity =999
        if (true_neg + false_pos)!=0:
            specificity = true_neg / (true_neg + false_pos)
        else:
            specificity = 999
        return sensitivity, specificity, accuracy  




def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):

    if model_name == "resnet18":
        """ Resnet18
        """
        # 导入模型结构
        model = models.resnet18(pretrained=True)
        # 加载预先下载好的预训练参数到resnet18
        
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        #model.load_state_dict(torch.load('39.pt'))
    if model_name == "resnet_6":
        """ Resnet18
        """
        # 导入模型结构
        model = models.resnet18(pretrained=True)
        # 加载预先下载好的预训练参数到resnet18
        
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        #model.maxpool=nn.MaxPool2d(kernel_size=5, stride=2, padding=1, dilation=1, ceil_mode=False)
        #model.layer1[1]=nn.Sequential()
        #model.layer2[1]=nn.Sequential()
        model.layer3[1]=nn.Sequential()
        model.layer4[1]=nn.Sequential()
        model.fc = nn.Linear(num_ftrs, num_classes)
        #model.load_state_dict(torch.load('39.pt'))
    if model_name == "resnet_3":
        """ Resnet18
        """
        # 导入模型结构
        model = models.resnet18(pretrained=True)
        # 加载预先下载好的预训练参数到resnet18
        
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.maxpool=nn.MaxPool2d(kernel_size=5, stride=2, padding=1, dilation=1, ceil_mode=False)
        model.layer1=nn.Sequential()
        model.layer2[1]=nn.Sequential()
        model.layer3[1]=nn.Sequential()
        model.layer4[1]=nn.Sequential()
        model.fc = nn.Linear(num_ftrs, num_classes)
    if model_name == "resnet_2":
        """ Resnet18
        """
        # 导入模型结构
        model = models.resnet18(pretrained=True)
        # 加载预先下载好的预训练参数到resnet18
        
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.conv1=nn.Conv2d(3, 128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.bn1=nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        model.layer1=nn.Sequential()
        model.layer2=nn.Sequential()
        model.layer3[1]=nn.Sequential()
        model.layer4[1]=nn.Sequential()
        model.fc = nn.Linear(num_ftrs, num_classes)
        #model.load_state_dict(torch.load('39.pt'))
    if model_name == "resnet_4":
        """ Resnet18
        """
        # 导入模型结构
        model = models.resnet18(pretrained=True)
        # 加载预先下载好的预训练参数到resnet18
        
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        #model.maxpool=nn.MaxPool2d(kernel_size=5, stride=2, padding=1, dilation=1, ceil_mode=False)
        model.layer1[1]=nn.Sequential()
        model.layer2[1]=nn.Sequential()
        model.layer3[1]=nn.Sequential()
        model.layer4[1]=nn.Sequential()
        model.fc = nn.Linear(num_ftrs, num_classes)
        #model.load_state_dict(torch.load('39.pt'))

    if model_name == "resnet34":
        model = models.resnet34(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        #model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


        model.fc = nn.Linear(num_ftrs, 2)
        #model.load_state_dict(torch.load('1_model_classification_trained.pt'))
        #model.fc = nn.Linear(num_ftrs, num_classes)

        
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
    if model_name == "resnet152":
        model = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
   
    elif model_name == "vgg11":
        model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11_bn', pretrained=True)
        model.classifier[6] = nn.Linear(4096,num_classes)
    elif model_name == "frcnn":
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif model_name == "lenet":
        model = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes) 
    return model



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


    

# =============================================================================
# Adjsut image to suitable size and format
# transforms
# =============================================================================

def adjust_img(img, phase,lab):
    
    ### normalizing according to current data
    img = np.nan_to_num(img)
    img = img - img.mean()
    if img.max()!=0:
     img = img / img.max()
    img = np.nan_to_num(img)
    
    
    
    if phase == 'train':
        #if lab==1:
          if args['augm_hflip']:
            if 0.3 > random.random():
                img = np.copy(np.flip(img, axis=1))
          if args['augm_vflip']:
            if 0 > random.random():
                img = np.copy(np.flip(img, axis=0))
          if args['augm_crop']:
            if 0.3 > random.random():
                h, w, _  = img.shape 
                minimum = np.minimum(h, w)
                margin = int(minimum*0.1)
                margin_final = int(random.randint(0, margin)/2)
                img = img[margin_final:(h-margin_final),margin_final:(w-margin_final)]
          if args['augm_rotate']:
            if 0.3 > random.random():
                x = random.randint(-25, 25)
                img = ndimage.rotate(img, x, reshape=False)

    #h, w , _= img.shape 
    h, w = img.shape 
    #if h<w:
        #ss=np.zeros((int((w-h)/2),w))
        #img = np.row_stack((ss, img,ss))
    #else:
    #    ss=np.zeros((h,int((h-w)/2)))
        #img = np.column_stack((ss, img,ss))
    #minimum = np.minimum(h, w)
    #maximum = np.maximum(h, w)
    #s=int(2.1*h/10)
    #g=int(2*h/5)
    #p=int(w/12)
    #q=int(w/10)
    #margin = int((maximum - minimum)/2)
    #if h > w:
        #img = img[margin:(h-margin),:]
    #if h < w:
        #img = img[:,margin:(w-margin)]
    #img = img[:minimum, :minimum]
    #img=img[s:h-g,p:w-q]
    #img = resize(img, (args['input_size'], args['input_size']))
    img = np.dstack((img,img, img))
    img = np.transpose(img, (2,0,1))    
    img = torch.from_numpy(img)
       
    return img
