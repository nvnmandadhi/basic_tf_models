import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential

((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

trainX = trainX / 255
testX = testX / 255


def train():
    if train_model:
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=(['accuracy']))
        model.fit(trainX, trainY, epochs=10)
        model.save('fashion_mnist')


if tf.keras.models.load_model('fashion_mnist') is not None:
    model = tf.keras.models.load_model('fashion_mnist')
    train_model = False
    train()
else:
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(124, activation=tf.nn.relu))
    model.add(Dense(10, activation=tf.nn.sigmoid))

predictions = model.predict(testX)
print(np.argmax(predictions[0]))
print(testY[0])
