import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import glob
import os
H = W = 256

def create_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)
def sharpen(img, sigma=5):    
    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    return usm

def create_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)
    
def LoadImage(img_path,color,filter_use,):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(H , W ))

    if filter_use == 'bilateral':
        img = cv2.bilateralFilter(img,7, 75, 75)
    elif filter_use == 'median':
        img = cv2.medianBlur(img, 5)
    elif filter_use == 'sharp':
        img = sharpen(img)
    elif filter_use == 'reduce':
        # img = reduce(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        img= img& int("11111100",2) # remain 64 bit
        img = cv2.cvtColor(img,cv2.COLOR_LAB2BGR)
    elif filter_use == 'reduce1':
        # img = reduce(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        img= img& int("11111110",2) # remain 64 bit
        img = cv2.cvtColor(img,cv2.COLOR_LAB2BGR)

        
    if color == 'cie':
        img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    img = img/255.0
    return img 

def LoadMask(mask_path):
    mask = cv2.imread(mask_path,0)
    mask = cv2.resize(mask,(H , W))
    return mask

def get_onehot(im,n_classes):
    one_hot = np.zeros((im.shape[0], im.shape[1], n_classes))
    for i in range(n_classes):
        one_hot[:,:,i] = (im == i).astype(int)
    return one_hot

def DataGenerator(path,mask_path, classes, batch_size=4,colour='rgb',prefilter='no'):
    files = path
    mask_files = mask_path

    while True:
        for i in range(0, len(files), batch_size):
            batch_files = files[i : i+batch_size]
            batch_masks = mask_files[i:i+batch_size]
            imgs=[]
            segs=[]
            for file in batch_files:
                image = LoadImage(file,colour,prefilter)
                imgs.append(image)

            for mask_file in batch_masks:
                mask = LoadMask(mask_file)
                labels = get_onehot(mask,classes)
                segs.append(labels)

            yield np.array(imgs), np.array(segs)
            
def equalize_clahe_color_lab(img):
    """equalize the image splitting after conversion to lab and applying clahe"""
    cla = cv2.createCLAHE(clipLimit=2)
    L,a,b = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2LAB))
    eq_L = cla.apply(L)
    eq_image = cv2.cvtColor(cv2.merge([eq_L,a,b]),cv2.COLOR_Lab2BGR)
    return eq_image
