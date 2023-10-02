"""
custom ImgDataset
"""

from torch.utils.data import Dataset
import utilz as ut
import cv2
from PIL import Image
import pydicom
from torchvision import transforms as tfs
import numpy as np
from skimage.transform import resize

class ImgDataset(Dataset):

    def __init__(self, img, lab, phase):

        self.img = img
        self.lab = lab
        self.phase = phase



    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
         
        lab = self.lab[idx]
        file=self.img[idx]
        img = cv2.imread(self.img[idx])
        
        #ds=pydicom.dcmread(self.img[idx])
        #img= ds.pixel_array
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #img = img.unsqueeze(dim=0) 
        
        
        
        img = ut.adjust_img(img, self.phase,lab).float()
        sample = {'img': img, 'lab': lab,'file':file}      

        return sample
    
    
    