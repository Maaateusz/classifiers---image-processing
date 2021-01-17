#%%

import time
start_time = time.time()

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Sequential

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

model_name = "projekt_2_cnn_VGG16_2"
model_path = 'G:/Projects/Datasets/models/Intel-Image-Classification/' + model_name +'.h5'
folder = 'G:/Projects/Datasets/Intel-Image-Classification/'
print("Data path: ", folder)
print("Model save path: ", model_path)

batch_size = 32
img_height = 150
img_width = 150
validation_split = 0.3
epochs = 15
with_cache = False

import pathlib
data_dir = pathlib.Path(folder)

print("___Training dataset___")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  pathlib.Path.joinpath(data_dir, "train/"),
  image_size=(img_height, img_width),
  batch_size=batch_size
)
print("▔▔▔▔▔▔▔▔▔▔▔▔▔")

print("___Validation dataset___")
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  pathlib.Path.joinpath(data_dir, "test/"),
  validation_split=validation_split,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)
print("▔▔▔▔▔▔▔▔▔▔▔▔▔")

print("___Test dataset___")
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  pathlib.Path.joinpath(data_dir, "test/"),
  validation_split=validation_split,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)
print("▔▔▔▔▔▔▔▔▔▔▔▔▔")

class_names = train_ds.class_names
print("Class names: ")
print(class_names)

for image_batch, labels_batch in train_ds:
  print("Shape:")
  print(image_batch.shape)
  print(labels_batch.shape)
  break

if(with_cache):
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  print(AUTOTUNE)
  train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

data_augmentation = keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width,3)),
  layers.experimental.preprocessing.RandomRotation(0.1),
  layers.experimental.preprocessing.RandomZoom(0.1),
])

time_start = time.time()

# ------------------------- VGG Model -------------------------

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

model_VGG = VGG16(
  weights='imagenet',
  include_top = False,
  input_shape=(img_height, img_width, 3)
)
model_VGG.summary()

model = Sequential([
  model_VGG,
  
  keras.layers.Flatten(),    

  keras.layers.Dense(128, activation='relu'),    
  layers.Dropout(0.4),

  layers.Dense(len(class_names))
])

model.compile(
  optimizer=optimizers.Nadam(lr=2e-5),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.summary()

print("----------------------------- Train -----------------------------")
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
print("----------------------------------------------------------")

model.save(model_path)

print("----------------------------- Evaluate -----------------------------")
test_scores = model.evaluate(test_ds)
print(test_scores)

labels = []
predicions = []
test_accuracy = tf.keras.metrics.Accuracy()
labelsF = []
predicionsF = []
for (x, y) in test_ds:
    logits = model(x, training=False)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    predicions.append(prediction)
    labels.append(y)
    test_accuracy(prediction, y)
print("   || Test set accuracy: {:.3%} ".format(test_accuracy.result()))

for y in labels:
    for y2 in y:
        labelsF.append(y2)
    
for y in predicions:
    for y2 in y:
        predicionsF.append(y2)

predictions = model.predict(test_ds)
predicted_class = np.argmax(predictions, axis=-1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labelsF, predicionsF)

time_end = time.time() - time_start
print("Czas Score: %.2f sec" % (time_end))
print("Confusion matrix: ")
print(cm)

print("----------------------------------------------------------")

print("----------------------------- Visualize results -----------------------------")
def print_plot():
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  plt.figure(figsize=(7,7))
  epochs_range = range(epochs)
  plt.plot(epochs_range, acc, '.-r', label='Training Accuracy')
  plt.plot(epochs_range, val_acc, '.-g', label='Validation Accuracy')
  plt.legend(loc='best')
  plt.title('Training and Validation Accuracy')

  plt.figure(figsize=(7,7))
  plt.plot(epochs_range, loss, '.-r', label='Training Loss')
  plt.plot(epochs_range, val_loss, '.-g', label='Validation Loss')
  plt.legend(loc='best')
  plt.title('Training and Validation Loss')
  plt.show()

print_plot()

from tensorflow.keras.utils import plot_model
plot_model(model, model_name+'_info.png', show_shapes=True)

import seaborn as sns
plt.figure(figsize=(7, 7))
sns.set(font_scale=1.1)
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt="d", ax = ax, annot_kws={"size": 14})
ax.set_xlabel('Predicted');ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)
plt.show()
print("----------------------------------------------------------")

end_time = time.time() - start_time
print("\nCzas: %.2f sec" % (end_time))