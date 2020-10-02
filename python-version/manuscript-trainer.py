import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import glob
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import callbacks 
from tensorflow import keras
from datetime import datetime
import tensorboard



class Symbol:
    symbol_class = "unknown_symbol"
    filepath="unknown"
    def __init__(self, filename):
        self.filename = os.path.basename(filename)
        splitted_by_slashes=filename.split("/")
        self.symbol_class=splitted_by_slashes[len(splitted_by_slashes)-2]



if __name__ == "__main__":
    #This first section mostly follows the tutorial at https://www.tensorflow.org/tutorials/images/classification
    data_dir = "/Users/ivy/Desktop/Senior_Seminar/HOMUS-Bitmap-Without-Git"
    image_count = len(list(glob.glob(f'{data_dir}/*/*.png')))
    print(image_count)

    batch_size = 128
    img_height = 100 #100 cleanly goes into our image resolution, so it just has to downscale by 3x
    img_width = 100

    #Set up training data
    val_split = 0.2

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=val_split,
    subset="training",
    seed=823492389,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical")

    #Set up testing data
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=val_split,
    subset="validation",
    seed=823492389,
    image_size=(img_height, img_width),
    batch_size=batch_size, 
    color_mode='rgb',
    label_mode="categorical")

    class_names = train_ds.class_names
    print(class_names)

    num_classes = len(train_ds.class_names)

    #   Normalize data
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))


    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")


    #Set up model

    model = tf.keras.Sequential()
    model.add(layers.experimental.preprocessing.Rescaling((1./255),input_shape=(100, 100, 3)))
    model.add(layers.Conv2D(64, (3,3), activation='relu',input_shape=(100, 100, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (5,5)))
    model.add(layers.MaxPooling2D(pool_size=(3,3)))
    model.add(layers.Dense(64))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='softmax'))

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


    model.summary()

    earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  
                                        mode ="min", patience = 7,  
                                        restore_best_weights = True) 

    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


    history=model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[earlystopping, tensorboard_callback]
    )
