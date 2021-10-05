# -*- coding:utf-8 -*-
from tkinter import *
from PIL import Image
# from PredictHandwrittenDigits import predict_digit
import tensorflow as tf
import keras
print(keras.__version__)
import numpy as np

model = tf.keras.models.load_model("mnist.h5")

def predict_digit(img):
    img = img.resize((28,28))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img/255.0
    """predict"""
    res = model.predict([img])[0]
    return np.argmax(res), max(res)


def main():
    img = Image.open('test-1.png')
    # img.show()
    digit, likelihood =predict_digit(img)
    print("digit: ", digit)
    print("likelihood: ", '%-10.2f' % (likelihood*100) + '%')


if __name__ == '__main__':
    main()



