import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

samples = 20
epochs = 200
# np.random.seed()
# train_x = np.linspace(0, 20, samples)
# train_y = 4 * train_x + 2 * np.random.randn(samples)

os.chdir("/Users/naveenmandadhi/Desktop/apps/tf")
data = pd.read_excel('data.xls', names=['train_x', 'train_y'])

train_x = data['train_x']
train_y = data['train_y']

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(np.random.randn(), name="weights")
B = tf.Variable(np.random.randn(), name="bias")

prediction = tf.add(tf.multiply(W, X), B)
cost = tf.reduce_sum((prediction - Y) ** 2) / (2 * samples)

rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for x, y in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        w = sess.run(W)
        b = sess.run(B)
        c = sess.run(optimizer, feed_dict={X: x, Y: y})
        if epoch % 20:
            print(f'{epoch:4d} {w:.4f} {b:.4f}')

    plt.plot(train_x, train_y, 'o')
    plt.plot(train_x, w * train_x + b)
    plt.show()
