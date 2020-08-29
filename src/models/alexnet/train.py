import datetime
import glob
import os
import pathlib
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from src.callbacks.last_model_manager import LastModelManager
from src.models.alexnet.model import AlexNet
from src.utilities.image import (
    crop_center,
    fancy_pca,
    resize_image_keep_aspect_ratio,
    subtract_mean,
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


def get_epoch_from_last_model_path(last_model_path: str) -> int:
    match = re.search(r"last_model_(\d{2})", last_model_path)
    if match is not None:
        epoch = int(match.group(1))
        return epoch
    return 0


def get_last_model_path(base_dir: str):
    paths = glob.glob(os.path.join(base_dir, "last_model_*"))
    if paths:
        last_model_path = paths[0]
        return last_model_path
    return None


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

    scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=10, min_lr=0.00001
    )

    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir)

    base_dir = ".\\models\\alexnet"
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

    last_model_path = get_last_model_path(base_dir)

    model = AlexNet(num_classes)
    initial_epoch = 0
    if last_model_path:
        initial_epoch = get_epoch_from_last_model_path(last_model_path)
        model = tf.keras.models.load_model(last_model_path)

    model.fit(
        train_ds,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
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
