import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPool2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD

from src.utilities.image import crop_center, fancy_pca, resize_image_keep_aspect_ratio


class AlexNet(Sequential):
    def __init__(self):
        super().__init__()
        self.add(
            Conv2D(
                96,
                11,
                strides=4,
                padding="same",
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="zeros",
                input_shape=(224, 224, 3),
            )
        )
        self.add(
            Lambda(
                lambda X: tf.nn.lrn(X, bias=2, depth_radius=5, alpha=0.0001, beta=0.75)
            )
        )
        self.add(MaxPool2D(pool_size=3, strides=2, padding="valid"))

        self.add(
            Conv2D(
                256,
                5,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="ones",
            )
        )
        self.add(
            Lambda(
                lambda X: tf.nn.lrn(X, bias=2, depth_radius=5, alpha=0.0001, beta=0.75)
            )
        )
        self.add(MaxPool2D(pool_size=3, strides=2, padding="valid"))

        self.add(
            Conv2D(
                384,
                3,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="zeros",
            )
        )
        self.add(
            Conv2D(
                384,
                3,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="ones",
            )
        )
        self.add(
            Conv2D(
                256,
                3,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="ones",
            )
        )
        self.add(MaxPool2D(pool_size=3, strides=2, padding="valid"))

        self.add(Flatten())
        self.add(
            Dense(
                4096,
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="ones",
            )
        )
        self.add(Dropout(0.5))
        self.add(
            Dense(
                4096,
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="ones",
            )
        )
        self.add(Dropout(0.5))
        self.add(
            Dense(
                1000,
                activation="softmax",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="zeros",
            )
        )

        optimizer = SGD(learning_rate=0.01, momentum=0.9, decay=0.0005)
        self.compile(
            optimizer=optimizer, loss=categorical_crossentropy, metrics=["accuracy"]
        )


def get_mappings():
    mappings = {}
    f = open("F:\\caffe_ilsvrc12.tar\\caffe_ilsvrc12\\synset_words.txt")
    for line in f:
        splits = line.split(" ", 1)
        folder_name = splits[0]
        class_names = splits[1]
        class_name = class_names.split(",", 1)[0]
        class_name = class_name.rstrip()
        mappings[folder_name] = class_name
    f.close()
    return mappings


def show_batch(image_batch, label_batch):
    mappings = get_mappings()
    plt.figure(figsize=(10, 10))
    for n in range(25):
        plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        name = mappings[CLASS_NAMES[label_batch[n] == 1][0]]
        plt.title(name)
        plt.axis("off")
    plt.show()


if __name__ == "__main__":

    data_dir = pathlib.Path("F:\\ILSVRC2012_img_train")
    CLASS_NAMES = np.array([item.name for item in data_dir.glob("*")])
    NUM_SAMPLES = 1281167
    BATCH_SIZE = 64

    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2] == CLASS_NAMES

    def decode_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img

    def augment(img):
        img = resize_image_keep_aspect_ratio(img)
        img = crop_center(img)
        img = tf.image.random_crop(img, size=[224, 224, 3])
        img = tf.image.random_flip_left_right(img)
        img = fancy_pca(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def process_path(file_path: str):
        label = get_label(file_path)
        image = decode_image(file_path)
        image = augment(image)
        return image, label

    #  repeat -> shuffle -> map -> batch -> batch-wise map -> prefetch
    dataset = (
        tf.data.Dataset.list_files(str(data_dir / "*/*"))
        .repeat()
        .shuffle(NUM_SAMPLES)
        .map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    model = AlexNet()
    scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=10, min_lr=0.00001
    )

    model.fit(
        dataset,
        epochs=90,
        steps_per_epoch=NUM_SAMPLES // BATCH_SIZE,
        callbacks=[scheduler],
    )
