import os

import cv2
import numpy as np
import tensorflow as tf

from random import shuffle


DIR = "dataset"
CATEGORIES = 2


def getDataset(IMG_SIZE, normalize = False):
    
    training_data = []
    
    for category in range(CATEGORIES):
        for file in os.listdir(f"{DIR}/{category}"):

            try:
                img_path = f"{DIR}/{category}/{file}"
                img = cv2.imread(img_path, 1)

                new_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_img, category])
            except Exception as e:
                print(f"[ERROR]: " + str(e))

    shuffle(training_data)

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    y = np.array(y).reshape(-1, 1)

    if normalize:
        X = tf.keras.utils.normalize(X, axis=1)
        y = tf.keras.utils.normalize(y, axis=1)

    return X, y

