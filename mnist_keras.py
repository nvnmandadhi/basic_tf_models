import tensorflow as tf
from tensorflow.python.keras.layers import Flatten, Activation, Dense
from tensorflow.python.keras.models import Sequential

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(x_train.shape[0], 784, 1)
x_test = x_test.reshape(x_test.shape[0], 784, 1)


def build_model():
    if not tf.keras.models.load_model('mnist_keras') is None:
        return tf.keras.models.load_model('mnist_keras')
    else:
        model1 = Sequential()
        model1.add(Flatten(input_shape=(784, 1)))
        model1.add(Dense(128))
        model1.add(Activation(tf.nn.relu))
        model1.add(Dense(10, activation=tf.nn.softmax))
        return model1


model = build_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4)

model.save('mnist_keras')
