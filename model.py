from keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tqdm import tqdm
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19

def fcn(input_shape , classes = 2, fcn8 = False, fcn16 = False,fcn32=False):
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    pool5 = vgg.get_layer('block5_pool').output 
    pool4 = vgg.get_layer('block4_pool').output
    pool3 = vgg.get_layer('block3_pool').output

    conv_6 = Conv2D(4096, (7, 7), activation='relu', padding='same', name="conv_6")(pool5)
    conv_6 =Dropout(0.5)(conv_6)
    conv_7 = Conv2D(4096, (1, 1), activation='relu', padding='same', name="conv_7")(conv_6)
    conv_7 = Dropout(0.5)(conv_7)

    conv_7 = (Conv2D( classes,(1,1),padding="same",activation=None))(conv_7)

    conv_7 = Conv2DTranspose(classes,kernel_size=(4,4),strides=(2,2),use_bias=False, padding='same', activation='relu')(conv_7)

    conv_8 = Conv2D(classes, (1, 1), padding="same",activation=None,name="conv_8")(pool4)
    
    add_1= Add()([conv_7,conv_8])

    o= Conv2DTranspose(classes,kernel_size=(4,4),strides=(2,2),use_bias=False,padding='same', activation='relu')(add_1)

    conv_9 = Conv2D(classes, (1, 1), padding='same' ,name="conv_9")(pool3)

    add_2 = Add( name="seg_feats")([conv_9,o])

    deconv_9 = Conv2DTranspose(classes, kernel_size=(16,16), strides=(8,8),padding='same')(add_2)
    
    if fcn8 :
        output_layer = Activation('softmax')(deconv_9)
    elif fcn16 :
        deconv_10 = Conv2DTranspose(classes, kernel_size=(8,8), strides=(8,8))(add_1)
        output_layer = Activation('softmax')(deconv_10)
    else :
        deconv_11 = Conv2DTranspose(classes, kernel_size=(16,16), strides=(16,16))(conv_7)
        output_layer = Activation('softmax')(deconv_11)


    model = Model(inputs=vgg.input, outputs=output_layer)
    return model