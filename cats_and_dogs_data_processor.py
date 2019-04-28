import os
import pickle
import random

import cv2
import matplotlib
import numpy as np

matplotlib.use("TkAgg")

os.chdir("/Users/naveenmandadhi/Desktop/apps/tf/PetImages")
categories = ["Cat", "Dog"]


def get_data():
    training_data = []
    for category in categories:
        path = os.path.join(os.getcwd(), category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                array = cv2.resize(img_array, (50, 50))
                training_data.append([array, class_num])
            except Exception:
                pass
    random.shuffle(training_data)
    return training_data


def process_data():
    global data
    data = get_data()

    x = []
    y = []

    for features, label in data:
        x.append(features)
        y.append(label)

    x = np.array(x).reshape(-1, 50, 50, 1)

    pickle_out = open("X.pickle", "wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


process_data()

print(len(data))
