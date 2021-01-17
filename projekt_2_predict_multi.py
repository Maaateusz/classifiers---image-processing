import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from matplotlib.image import imread
from os import listdir, path
from keras.preprocessing.image import array_to_img
from random import randint

dataset_dir = 'G:/Projects/Datasets/Intel-Image-Classification/pred/'
loaded_images = list()
predicted = list()
img_arrays = list()
valid_images = [".jpg", ".png"]

batch_size = 32
img_height = 150
img_width = 150
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
max_img = 25

# model_name = "projekt_2_cnn_85pr"
# model_name = "projekt_2_cnn_VGG16_92pr"
model_name = "projekt_2_classic_nn"
model_path = 'G:/Projects/Datasets/models/Intel-Image-Classification/' + model_name +'.h5'
model = load_model(model_path)

def show_images(images, cols = 1, titles = None): 
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.savefig('predicted_multi.png', bbox_inches='tight')

def pred_images():
    for filename in listdir(dataset_dir):
        end = path.splitext(filename)[1]
        if end.lower() in valid_images:
            img = load_img(dataset_dir + filename, target_size=(img_height, img_width))
            loaded_images.append(img)
            # img_array = img_to_array(img)
            # img_array = tf.expand_dims(img_array, 0) 
            # img_arrays.append(img_array)
            # predictions = model.predict(img_array)
            # score = tf.nn.softmax(predictions[0])
            # print("____________________________")
            # print("|-> Image: %s, %s px" % (filename, img.size))
            # print("|______ Class: '{}', Acc: {:.2f}%".format(class_names[np.argmax(score)], 100 * np.max(score)))
            # predicted.append("{} | {} | {:.2f}%".format(filename, class_names[np.argmax(score)], 100 * np.max(score)))

def pred_images_x():
    filenames = listdir(dataset_dir)
    for i in range(20, 45):
        filename = filenames[i]
        end = path.splitext(filename)[1]
        if end.lower() in valid_images:
            img = load_img(dataset_dir + filename, target_size=(img_height, img_width))
            loaded_images.append(img)

#Plotting a random subset of our images
def show_images():
    f,ax = plt.subplots(5,5) 
    f.subplots_adjust(0,0,3,3)
    for i in range(0,5,1):
        for j in range(0,5,1):
            rnd_number = randint(0, len(loaded_images))
            ax[i,j].imshow(loaded_images[rnd_number])
            img_array = img_to_array(loaded_images[rnd_number])
            img_array = tf.expand_dims(img_array, 0) 
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            score_ = ("Class: '{}', Acc: {:.2f}%".format(class_names[np.argmax(score)], 100 * np.max(score)))
            ax[i,j].set_title(score_)
            ax[i,j].axis('off')
    plt.savefig('projekt_2/predicted_multi.png', bbox_inches='tight')
    
def show_images_x():
    c = 0
    f,ax = plt.subplots(5,5) 
    f.subplots_adjust(0,0,3,3)
    for i in range(0,5,1):
        for j in range(0,5,1):
            ax[i,j].imshow(loaded_images[c])
            img_array = img_to_array(loaded_images[c])
            c = c + 1
            img_array = tf.expand_dims(img_array, 0) 
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            score_ = ("Class: '{}', Acc: {:.2f}%".format(class_names[np.argmax(score)], 100 * np.max(score)))
            ax[i,j].set_title(score_)
            ax[i,j].axis('off')
    plt.savefig('projekt_2/predicted_multi.png', bbox_inches='tight')


pred_images_x()
# pred_images()
# show_images(loaded_images, 1, predicted)
# show_images()
show_images_x()
