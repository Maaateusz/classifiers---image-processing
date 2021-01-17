# %%

import time
start_time = time.time()

from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pandas as pd


img_height = 150
img_width = 150
validation_split = 0.2
folder_train = 'G:/Projects/Datasets/Intel-Image-Classification/train/'
folder_size = []
class_names = []

# https://medium.com/swlh/image-classification-with-k-nearest-neighbours-51b3a289280
def load_images(folder_path):
    image_dir = Path(folder_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    for fo in folders:
        class_names.append(fo.name)
    train_img = []
    for i, direc in enumerate(folders):
        count_in = 0
        for file in direc.iterdir():
            count_in += 1
            imgRaw = load_img(file)
            img = img_to_array(imgRaw)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img_pred = cv.resize(img, (img_height, img_width), interpolation=cv.INTER_AREA)
            img_pred = image.img_to_array(img_pred)
            img_pred = img_pred / 255
            train_img.append(img_pred)
        folder_size.append(count_in)
    X = np.array(train_img)
    return X

X = []
X = load_images(folder_train)
y = []
for i in range(len(folder_size)):
    l = np.full(folder_size[i], i)
    y = np.concatenate((y, l), axis=0)
print(folder_size)
print(class_names)
print(y)
print(len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=validation_split)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, random_state=42, test_size=validation_split)
print("X_train: "+str(X_train.shape))
print("X_test: "+str(X_test.shape))
print("X_val: "+str(X_val.shape))
print("y_train: "+str(y_train.shape))
print("y_test: "+str(y_test.shape))
print("y_val: "+str(y_val.shape))
print("------------------------------------ Reshape the image data into rows")

from builtins import range
from builtins import object

num_training = X_train.shape[0]
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = X_test.shape[0]
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

num_val = X_val.shape[0]
mask = list(range(num_val))
X_val = X_val[mask]
y_val = y_val[mask]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))

print("X_train: "+str(X_train.shape))
print("X_test: "+str(X_test.shape))
print("X_val: "+str(X_val.shape))
print("y_train: "+str(y_train.shape))
print("y_test: "+str(y_test.shape))
print("y_val: "+str(y_val.shape))

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

print("\nGaussian Naive Bayes")
time_start = time.time()
gnb = GaussianNB()
test_classes_predicted = gnb.fit(X_train, y_train).predict(X_val)
score = metrics.accuracy_score(y_val, test_classes_predicted) * 100
print("Accuracy:", score, "%")
time_end = time.time() - time_start
print("Czas Score: %.2f sec" % (time_end))
cm = confusion_matrix(y_val, test_classes_predicted)
print(cm)

plt.figure(figsize=(7, 7))
sns.set(font_scale=1.4)
sns.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 16})
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

end_time = time.time() - start_time
print("\nCzas: %.2f sec" % (end_time))