import csv
import os
from glob import glob
from shutil import copyfile
from random import shuffle, seed
import splitfolders
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from ultralytics import YOLO
import pandas as pd
import numpy as np
from IPython.display import display, Image
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

csv_file_path = os.path.join(os.getcwd(), 'driver_imgs_list.csv')

data = {}  # Dictionary to store data
csv_file_path = os.path.join(os.getcwd(), 'driver_imgs_list.csv')

with open(csv_file_path, 'r') as file:
    read_file = csv.reader(file)
    read_file = list(read_file)
    
    # Loop through the rows, skipping the header
    for row in read_file[1:]:
        key = row[1]  # Class label (e.g., 'c0', 'c1')
        img = row[2]  # Image file name (e.g., 'img_10206.jpg')
        
        # If the class label exists in the dictionary, append the image
        if key in data:
            data[key].append(img)
        else:
            # If it's a new class label, initialize with the image
            data[key] = [img]
classes_list = list(data.keys())
dataset_folder = 'imgs'
train_dir = os.path.join(dataset_folder, 'train')
test_dir = os.path.join(dataset_folder,'test')
dataset_small_folder_path = 'smallest'
subfolders = classes_list
if os.path.exists(dataset_small_folder_path):
    for root, dirs, files in os.walk(dataset_small_folder_path, topdown = False):
        for name in files:
            file_path = os.path.join(root, name)
            os.remove(file_path)
        for name in dirs:
            dir_path = os.path.join(root, name)
            os.rmdir(dir_path)

for folder in subfolders:
    subfolder_path = os.path.join(dataset_small_folder_path, folder)
    os.makedirs(subfolder_path)

for clas, images in data.items():
    length = len(images)
    seed(42)
    shuffle(images)
    for image in images[:int(length*0.7)]:
        source = os.path.join(dataset_folder, 'train/', clas, image)
        #print(source)                     
        destination = os.path.join(dataset_small_folder_path, clas, image)
        copyfile(source, destination)
'''print(dataset_small_folder_path)
for subfolder in subfolders:
    subfolder_path = os.path.join(dataset_small_folder_path, subfolder)
    print("Number of images for each class: ", subfolder, "->", len(os.listdir(subfolder_path)))'''
small_dataset = {}
for subfolder in os.listdir(dataset_small_folder_path):
    small_dataset[subfolder] = os.listdir(os.path.join(dataset_small_folder_path, subfolder))
def remove_directory(path):
    for root, dirs, files in os.walk(path, topdown = False):
        for name in files:
            file_path = os.path.join(root, name)
            os.remove(file_path)
        for name in dirs:
            dir_path = os.path.join(root, name)
            os.rmdir(dir_path)
    os.rmdir(path)
def create_directories(paths, subfolders):
    for path in paths:
        if os.path.exists(path):
            remove_directory(path)
        
        for folder in subfolders:
            subfolder_path = os.path.join(path, folder)
            os.makedirs(subfolder_path)
paths = ['cleaned_dataset/train',
         'cleaned_dataset/val',
        'cleaned_dataset/test']
subfolders = classes_list
create_directories(paths, subfolders)
split_size = [0.8, 0.1]
for clas, images in small_dataset.items():
    # print(len(images))
    train_size = int(split_size[0]*len(images))
    # print("Train size: ", train_size)
    
    test_size = int(split_size[1]*len(images))
    #print("Test size: ", test_size)
    
    train_images = images[:train_size]
    # print("Train Images Length", len(train_images))
    
    val_images = images[train_size: train_size + test_size]
    # print("Val Images Length", len(val_images))

    test_images = images[train_size + test_size:]
    # print("Test Images Length", len(test_images))

    for image in train_images:
        source = os.path.join(train_dir, clas, image)
        # print(os.path.exists(source))
        dest = os.path.join(paths[0], clas, image)
        copyfile(source, dest)
    
    for image in val_images:
        source = os.path.join(train_dir, clas, image)
        dest = os.path.join(paths[1], clas, image)
        copyfile(source, dest)
    
    for image in test_images:
        source = os.path.join(train_dir, clas, image)
        dest = os.path.join(paths[2], clas, image)
        copyfile(source, dest)
parent_dir = 'cleaned_dataset'
train_dir = os.path.join(parent_dir,'train')
val_dir = os.path.join(parent_dir,'val')
test_dir = os.path.join(parent_dir,'test')

#Image data generator
def imagedatageneration(train_dir, val_dir, test_dir, target_size = (256, 256), batch_size = 64):
    train_datagen = ImageDataGenerator(rescale = 1.0 / 255,
                                       rotation_range = 30,
                                       width_shift_range = 0.1,
                                       height_shift_range = 0.1,
                                       zoom_range = 0.1,
                                       shear_range = 0.1,
                                       fill_mode = "nearest"
                                      )
    train_generator = train_datagen.flow_from_directory(
                                                            train_dir,
                                                            target_size = target_size,
                                                            class_mode = 'categorical',
                                                            shuffle = True,
                                                            batch_size = batch_size
                                                        )
    val_datagen = ImageDataGenerator(rescale = 1.0 / 255)
    
    val_generator = val_datagen.flow_from_directory(
                                                        val_dir,
                                                        target_size = target_size,
                                                        class_mode = 'categorical',
                                                        shuffle = True,
                                                        batch_size = batch_size
                                                    )
    test_datagen = ImageDataGenerator(rescale = 1.0/255)
    test_generator = test_datagen.flow_from_directory(
                                                        test_dir,
                                                        target_size = target_size,
                                                        class_mode = 'categorical',
                                                        shuffle = False,
                                                        batch_size = 1
                                                      )
    return train_generator, val_generator, test_generator

es = EarlyStopping(monitor = "val_acc",
                    min_delta = 0.0001,
                    verbose=1,
                    patience = 5,
                    restore_best_weights = True,
                    baseline = None)
'''def train_val_plot(model, model_name):
    train_loss, train_acc, val_loss, val_acc = model.history['loss'], model.history['acc'], model.history['val_loss'], model.history['val_acc']
    
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('{} Model Accuracy'.format(model_name))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper left')
    plt.show()

    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('{} Model Loss'.format(model_name))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper left')
    plt.show() '''

train_generator, val_generator, test_generator = imagedatageneration(train_dir, val_dir, test_dir)
#VGG-16

pretrained_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (256, 256, 3))
pretrained_model.summary()
for layer in pretrained_model.layers[:-5]:
    layer.trainable = False
last_layer = pretrained_model.get_layer('block4_pool')
last_output = last_layer.output

x = Flatten()(last_output)
x = Dropout(0.2)(x)
x = Dense(128, activation = 'relu')(x)
x = BatchNormalization()(x)
x = Dense(256, activation = 'relu')(x)
x = BatchNormalization()(x)
x = Dense(10, activation = 'softmax')(x)


model3 = Model(pretrained_model.input, x)
model3.compile(optimizer = Adam(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = ['acc'])
model3.summary()
model3_history =  model3.fit(train_generator,
            epochs = 15,
            verbose = 1,
            validation_data = val_generator,
            callbacks = [es])
accuracy = model3.evaluate(test_generator)
model3.save("vgg16_model.keras")

print("Accuracy based on our VGG16 Model :- {:.2f}%".format(accuracy[1]*100))