import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import tensorflow as tf 
from tqdm import tqdm
from tensorflow.keras.utils import CustomObjectScope
from keras import backend as K
import sys
import os 
import pandas as pd 
import segmentation_models as sm
from tensorflow.keras.utils import to_categorical
import glob 
from keras.metrics import MeanIoU
from utils import *
if __name__ == "__main__":
    absoulate_path = os.path.abspath(__file__)
    fileDirectory = os.path.dirname(absoulate_path) # 資料夾路徑
    name_file_n = 'path'

    test_data = 'image data path'
    save_path = 'save path'
    n_classes =4
    H = W = 256
        
    with CustomObjectScope({ 'f1-score': sm.metrics.f1_score,'iou_score':sm.metrics.iou_score  }):
        model = tf.keras.models.load_model(name_file_n)
        
    """ Predict the mask """
    input_ = cv2.imread(test_data) 
    x = cv2.resize(input_,(H,W))
    colour = 'rgb'
    filter_use = 'no'

    if colour == 'cie':
        x = cv2.cvtColor(x,cv2.COLOR_BGR2LAB)
    if filter_use == 'bilateral':
        x = cv2.bilateralFilter(x,7, 75, 75)
    elif filter_use == 'sharp':
        x = sharpen(x,5)
    elif filter_use == 'reduce':
        x = cv2.cvtColor(x,cv2.COLOR_BGR2LAB)
        x= x& int("11111100",2) # remain 64 bit
        x = cv2.cvtColor(x,cv2.COLOR_LAB2BGR)
    elif filter_use == 'reduce1':
        x = cv2.cvtColor(x,cv2.COLOR_BGR2LAB)
        x= x& int("11111110",2) # remain 64 bit
        x = cv2.cvtColor(x,cv2.COLOR_LAB2BGR)

    x = x/255.

    """Predict image"""
    pred = model.predict(x.reshape(1,H,W,3))[0]
    seg = result_mask(pred,n_classes)
    seg = np.uint8(seg)*255
    cv2.imwrite(save_path,seg)