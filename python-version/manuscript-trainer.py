import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import glob
import matplotlib.pyplot as plt
from tensorflow.keras import layers


class Symbol:
    symbol_class = "unknown_symbol"
    filepath="unknown"
    def __init__(self, filename):
        self.filename = os.path.basename(filename)
        splitted_by_slashes=filename.split("/")
        self.symbol_class=splitted_by_slashes[len(splitted_by_slashes)-2]



if __name__ == "__main__":
    data_dir = "/Users/ivy/Desktop/Senior_Seminar/HOMUS-Bitmap-Without-Git"
    image_count = len(list(glob.glob(f'{data_dir}/*/*.png')))
    print(image_count)

    batch_size = 32
    img_height = 180
    img_width = 180

    #Set up training data
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='rgba')

    #Set up testing data
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='rgba')

    class_names = train_ds.class_names
    print(class_names)





    # Set up autotune
    # AUTOTUNE = tf.data.experimental.AUTOTUNE

    # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(train_ds.class_names)

    #Set up model

    model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 4)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu')
])

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.summary()

history = model.fit(train_ds, epochs=10, 
                    validation_data=(val_ds))
