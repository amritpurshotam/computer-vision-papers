import datetime
import os
import pathlib
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPool2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD

from src.callbacks.last_model_manager import LastModelManager
from src.utilities.image import (
    crop_center,
    fancy_pca,
    resize_image_keep_aspect_ratio,
    subtract_mean,
)


class AlexNet(Sequential):
    def __init__(self, num_class: int):
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
                num_class,
                activation="softmax",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="zeros",
            )
        )

        optimizer = SGD(learning_rate=0.01, momentum=0.9, decay=0.0005)
        self.compile(
            optimizer=optimizer, loss=categorical_crossentropy, metrics=["accuracy"]
        )


def get_label(file_path, class_names):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == class_names


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
    img = tf.cast(img, dtype=tf.float32)
    img = subtract_mean(img)
    return img


def process_path(file_path: str, class_names: np.ndarray):
    label = get_label(file_path, class_names)
    image = decode_image(file_path)
    image = augment(image)
    return image, label


def build_dataset(data_dir: Path, num_samples: int, batch_size: int):
    #  repeat -> shuffle -> map -> batch -> batch-wise map -> prefetch
    ds = (
        tf.data.Dataset.list_files(str(data_dir / "*/*"))
        .repeat()
        .shuffle(num_samples)
        .map(
            lambda file_path: process_path(file_path, CLASS_NAMES),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    return ds


if __name__ == "__main__":

    # train_dir = pathlib.Path("F:\\ILSVRC2012_img_train")
    # val_dir = pathlib.Path("F:\\ILSVRC2012_img_val")
    # TRAIN_NUM_SAMPLES = 1281167
    # VAL_NUM_SAMPLES = 50000
    # BATCH_SIZE = 128

    train_dir = pathlib.Path(
        "F:\\imagenette\\downloads\\extracted\\TAR_GZ.s3_fast-ai-imageclas_imagenette2-320UCCpEwzqA0gnKCPLEtLbfpgcbyr6Pc5xzNW4ATAFxV4.tgz\\imagenette2-320\\train"
    )
    val_dir = pathlib.Path(
        "F:\\imagenette\\downloads\\extracted\\TAR_GZ.s3_fast-ai-imageclas_imagenette2-320UCCpEwzqA0gnKCPLEtLbfpgcbyr6Pc5xzNW4ATAFxV4.tgz\\imagenette2-320\\val"
    )
    TRAIN_NUM_SAMPLES = 9469
    VAL_NUM_SAMPLES = 3925
    BATCH_SIZE = 128

    EPOCHS = 90
    CLASS_NAMES = np.array([item.name for item in train_dir.glob("*")])
    num_classes = CLASS_NAMES.shape[0]

    train_ds = build_dataset(train_dir, TRAIN_NUM_SAMPLES, BATCH_SIZE)
    val_ds = build_dataset(val_dir, VAL_NUM_SAMPLES, BATCH_SIZE)

    model = AlexNet(num_classes)
    scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=10, min_lr=0.00001
    )

    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir)

    base_dir = "./models/alexnet"
    last_model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(base_dir, "last_model_{epoch:02d}-{val_accuracy:.2f}"),
        save_best_only=False,
        save_weights_only=False,
        save_freq="epoch",
    )
    last_model_manager = LastModelManager(base_dir)

    best_model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(base_dir, "best_model"),
        save_best_only=True,
        save_weights_only=False,
        monitor="val_accuracy",
        mode="max",
    )

    model.fit(
        train_ds,
        epochs=EPOCHS,
        initial_epoch=0,
        steps_per_epoch=TRAIN_NUM_SAMPLES // BATCH_SIZE,
        validation_data=val_ds,
        validation_steps=VAL_NUM_SAMPLES // BATCH_SIZE,
        callbacks=[
            scheduler,
            tensorboard,
            last_model_checkpoint,
            best_model_checkpoint,
            last_model_manager,
        ],
    )

# todo mixed precision training
# todo wandb callback
# todo model checkpointing
# todo tfrecords
