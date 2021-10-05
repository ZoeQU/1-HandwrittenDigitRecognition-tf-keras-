# -*- coding:utf-8 -*-
import tensorflow as tf
mnist = tf.keras.datasets.mnist

from tensorflow import keras
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras import models
from keras import Sequential
from keras.utils import plot_model
from keras.optimizers import SGD

"""load data"""
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train.shape, Y_train.shape)

"""preprocess the data"""
img_rows, img_cols = 28, 28
X_train, X_test = X_train / 255.0, X_test / 255.0
Y_train = keras.utils.to_categorical(Y_train, num_classes=10)
Y_test = keras.utils.to_categorical(Y_test, num_classes=10)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

"""create the model"""
batch_size = 128
num_classes = 10
epochs = 20

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])
model.summary()
plot_model(model, "net.svg", show_shapes=True)
"""train model"""
hist = model.fit(X_train, Y_train, batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,
                 validation_data=(X_test, Y_test))
print("the model has successfully trained")

model.save('mnist.h5')
print('saving the model as mnist.h5')

"""evaluate the model"""
score = model.evaluate(X_test, Y_test, verbose=0)
print('test loss: ', score[0])
print('test accuracy: ', score[1])



