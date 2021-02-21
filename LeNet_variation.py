import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical 
from keras.models import load_model
import tensorflow as tf

# Loading the MNIST datasets
(x_train, labels_train), (x_test, labels_test) = mnist.load_data()

# Padding images to prevent shrinkage
x_train = np.pad(x_train,((0,0),(2,2),(2,2)))
x_test = np.pad(x_test,((0,0),(2,2),(2,2)))
# Converting to floating poitn
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255.0
x_test = x_test/255.0

# A one hot encoding
y_train = to_categorical(labels_train, 10) 
y_test = to_categorical(labels_test, 10)

# Expanding array
x_train=np.expand_dims(x_train,3)
x_test=np.expand_dims(x_test,3)

# Initialiasing the the network
LeNet5_var = Sequential()
# C1 & C2
LeNet5_var.add(Conv2D(filters= 32,kernel_size = (5,5),strides=(1,1),activation='relu',input_shape=(32,32,1)))
LeNet5_var.add(Conv2D(32,(5,5),strides=(1,1), use_bias=False))
# B1
LeNet5_var.add(BatchNormalization())
LeNet5_var.add(Activation('relu'))
# MXPL1
LeNet5_var.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
LeNet5_var.add(Dropout(0.25))
# C3 & C4
LeNet5_var.add(Conv2D(filters=64,kernel_size = (5,5),strides=(1,1),activation='relu'))
LeNet5_var.add(Conv2D(64,(3,3),strides=(1,1), use_bias=False))
# B2
LeNet5_var.add(BatchNormalization())
LeNet5_var.add(Activation('relu'))
# MXPL2
LeNet5_var.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
LeNet5_var.add(Dropout(0.25))
LeNet5_var.add(Flatten())
# FC1
LeNet5_var.add(tf.keras.layers.Dense(units=256,activation='relu'))
# B3
LeNet5_var.add(BatchNormalization())
LeNet5_var.add(Activation('relu'))
# FC2
LeNet5_var.add(tf.keras.layers.Dense(units=120,activation='relu'))
#B4
LeNet5_var.add(BatchNormalization())
LeNet5_var.add(Activation('relu'))
# FC3
LeNet5_var.add(tf.keras.layers.Dense(units=84,activation='relu'))
# B5
LeNet5_var.add(BatchNormalization())
LeNet5_var.add(Activation('relu'))
LeNet5_var.add(Dropout(0.25))
# FC4
LeNet5_var.add(Dense(units=10,activation='softmax'))

# NETWORK TRAINING
variable_learning_rate = ReduceLROnPlateau(monitor='loss', factor = 0.2, patience = 2)
LeNet5_var.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = LeNet5_var.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=20, batch_size=256, callbacks = [variable_learning_rate])
# DAVEING THE MODEL
LeNet5_var.save("network1230.h5")

# LOADING MODEL
LeNet5_var = load_model("network123.h5")

# TESTING
outputs=LeNet5_var.predict(x_test)
labels_predicted=np.argmax(outputs, axis=1) 
misclassified=sum(labels_predicted!=labels_test) 
print('Percentage misclassified = ',100*misclassified/labels_test.size)


