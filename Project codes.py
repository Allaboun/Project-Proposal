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


# 160000 images for the model to pick up the local patterns


# building the model:

from keras.optimizers import Adam
model = tf.keras.models.Sequential([
    # 1st conv
  tf.keras.layers.Conv2D(96, (11,11),strides=(4,4), activation='relu', input_shape=(64, 64, 3)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, strides=(2,2)),
    # 2nd conv
  tf.keras.layers.Conv2D(256, (11,11),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
     # 3rd conv
  tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
    # 4th conv
  tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
    # 5th Conv
  tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),
  # To Flatten layer
  tf.keras.layers.Flatten(),
  # To FC layer 1
  tf.keras.layers.Dense(4096, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  #To FC layer 2
  tf.keras.layers.Dense(4096, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1, activation='sigmoid')
  ])
model.compile(
    optimizer=Adam(lr=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
   )

#Model Training using model.fit:
hist = model.fit(train_generator,steps_per_epoch=128,epochs=50,validation_data=validation_generator,validation_steps=128)


# plot the training vs validation loss and accuracy graphs:
import matplotlib.pyplot as plt
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(len(acc))

# plot Training & Validation Accuracy:
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'yo', label='Validation Accuracy')
plt.title('Training & Validation Accuracy:')
plt.legend(loc='lower center')
plt.figure()
plt.show()

# plot Training & Validation Loss:
plt.plot(epochs, loss, 'ro', marker='>', label='Training Loss')
plt.plot(epochs, val_loss, 'bo', marker='*', label='Validation Loss')
#plt.plot(epochs, loss, 'ro', label='Training Loss')
#plt.plot(epochs, val_loss, 'bo', label='Validation Loss')
plt.title('Training & Validation Loss:')
plt.legend(loc='upper center')
plt.figure()
plt.show()


import numpy as np
from keras.preprocessing import image

# Testing, the follwing image is mine and it is copletly not form the dataset:
path = "/Users/ASUSvB/Desktop/archive/Dataset/Test/Male/333.png"

# Testing, the follwing image is Not from the dataset completly new:
#path = "/Users/ASUSvB/Desktop/archive/Dataset/Test/Female/555.jpg"

img = image.load_img(path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# making decision
images = np.vstack([x])
classes = model.predict(images, batch_size=1)
print(classes[0])
if classes[0]> 0.5:
    print("-It is very likely he is Male.    .البرنامج يقول أنه رجل-")   #showing the decision in English & Arabic
elif classes[0] <= 0.5:
    print("-It is very likely she is Female.    .البرنامج يقول أنها أنثى-")  #showing the decision in English & Arabic
else:
    print( "Error!")
    
# showing the tested image    
plt.imshow(img)
