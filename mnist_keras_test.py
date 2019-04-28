import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model

os.chdir("/Users/naveenmandadhi/Desktop/apps/tf")

model = load_model('mnist_keras')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_test / 255

x_test = x_test.reshape(x_test.shape[0], 784, 1)
pred = model.predict(x_test)
print(np.argmax(pred[4]))

x_test = x_test.reshape(x_test.shape[0], 28, 28)
cv2.imshow('image', x_test[4])
