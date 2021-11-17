# import necessary libraries: 
import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# preparing image data generator with minor data augmentations to keep the training time optimal:
train_datagen = ImageDataGenerator(rescale = 1./255,
	  rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


test_datagen = ImageDataGenerator( rescale = 1.0/255)

train_generator = train_datagen.flow_from_directory('/Users/ASUSvB/Desktop/archive/Dataset/Train',
                                                    batch_size =128 ,
                                                    class_mode = 'binary', 
                                                    target_size = (64, 64))    

validation_generator =  test_datagen.flow_from_directory('/Users/ASUSvB/Desktop/archive/Dataset/Validation',
                                                          batch_size  = 128,
                                                          class_mode  = 'binary', 
                                                          target_size = (64, 64))


