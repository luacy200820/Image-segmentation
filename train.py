from keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plot
import glob
import PIL
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
# from keras.layers.merge import concatenate
from PIL import Image
import matplotlib.pyplot as plot
import cv2
import glob 
import segmentation_models as sm
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
import os
import time
import shutil
from model import *
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from utils import *
import gc
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
H = W = 256

if __name__ == "__main__":
    absoulate_path = os.path.abspath(__file__)
    fileDirectory = os.path.dirname(absoulate_path)
    dataset = 'dataset17'
    opendata_path = r'path'
    x_train=sorted(glob.glob(opendata_path+'/images/*.png') )# image postition
    y_train=sorted(glob.glob(opendata_path+'/masks/*.png')) # mask postition
    print(len(x_train))

    # train_ratio = 0.80
    test_ratio = 0.10
    validation_ratio = 0.20

    '''split dataset'''
    x_train, x_val ,y_train,y_val= train_test_split(x_train, y_train, test_size=test_ratio,shuffle=True)

    height= width = 256
    classes = 4
    LR = 0.0001 #0.0001
    EPOCHS = 150
    Decay = 1e-06
    color_space = 'rgb'
    filter_use = 'no' #filterUse
    model_name = 'fcn8'
    batch_size = 4
    
    train_gen = DataGenerator(x_train,y_train, classes, batch_size=batch_size,colour=color_space,prefilter=filter_use)
    val_gen = DataGenerator(x_val,y_val, classes, batch_size=batch_size,colour=color_space,prefilter=filter_use )
    
    num_of_training_samples = len(x_train) 
    num_of_testing_samples = len(x_val)
    
    print(f"Train: {num_of_training_samples} ")
    print(f"Valid: {num_of_training_samples} ") 

    '''model '''
    input_shape = (width, height, 3)
    model = fcn(input_shape,classes=classes, fcn8=True)
    
    dice_loss = sm.losses.DiceLoss() 
    iou_loss = sm.losses.JaccardLoss()
    cross_loss = sm.losses.CategoricalCELoss()

    cce_dice_loss = cross_loss + dice_loss
    focal_loss = sm.losses.BinaryFocalLoss() if classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)    
     
    adam = Adam(lr=LR, decay=1e-06)
    model.compile(optimizer=adam , loss= 'categorical_crossentropy' , metrics=[sm.metrics.f1_score, sm.metrics.iou_score])    
    new_path = fileDirectory+'/result'
    create_dir(new_path)

    filepath = new_path+f"/model.h5"
    callbacks = [
        ModelCheckpoint(filepath, monitor='val_iou_score', verbose=1, save_best_only=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min'),
        EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=False, mode='min')
    ]
    """Train model"""
    hist = model.fit   (
        train_gen, 
        epochs=EPOCHS, 
        steps_per_epoch=num_of_training_samples//batch_size,
        validation_data=val_gen, validation_steps=num_of_testing_samples//batch_size,
        callbacks=callbacks)

    loss = hist.history["val_loss"]
    acc = hist.history["val_iou_score"] #accuracy

    plot.figure(figsize=(12, 6))
    plot.subplot(211)
    plot.title("Val. Loss")
    plot.plot(loss)
    plot.xlabel("Epoch")
    plot.ylabel("Loss")

    plot.subplot(212)
    plot.title("Val. IOU")
    plot.plot(acc)
    plot.xlabel("Epoch")
    plot.ylabel("IOU")

    plot.tight_layout()
    
    plot.savefig(new_path+f"/learn.png", dpi=150)

    del model
    del train_gen
    del val_gen
    gc.collect()