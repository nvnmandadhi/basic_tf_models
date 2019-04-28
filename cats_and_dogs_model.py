import os
import pickle

import tensorflow as tf

os.chdir("/Users/naveenmandadhi/Desktop/apps/tf/PetImages/model")

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

X = X / 255.0

model = tf.keras.models.load_model('convo_model')

# model = Sequential()
#
# model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(256, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
#
# model.add(Dense(64))
#
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

model.fit(X, y, batch_size=100, epochs=3, validation_split=0.3)

model.save("convo_model")
