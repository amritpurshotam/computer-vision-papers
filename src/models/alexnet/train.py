import os
import pathlib
from pathlib import Path

import click
import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from wandb.keras import WandbCallback

from src.callbacks.helper import get_best_model_checkpoint, get_last_model_checkpoint
from src.callbacks.last_model_manager import LastModelManager, get_latest_trained_model
from src.models.alexnet.model import get_alexnet_model
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


def build_dataset(
    data_dir: Path, num_samples: int, batch_size: int, class_names: np.ndarray
):
    #  repeat -> shuffle -> map -> batch -> batch-wise map -> prefetch
    ds = (
        tf.data.Dataset.list_files(str(data_dir / "*/*"))
        .repeat()
        .shuffle(num_samples)
        .map(
            lambda file_path: process_path(file_path, class_names),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    return ds


@click.command()
@click.option(
    "--train_dir", type=str, required=True, help="Path to training set directory"
)
@click.option(
    "--val_dir", type=str, required=True, help="Path to validation set directory"
)
@click.option(
    "--train_num_samples",
    type=int,
    required=True,
    help="Number of samples in training set",
)
@click.option(
    "--val_num_samples",
    type=int,
    required=True,
    help="Number of samples in validation set",
)
@click.option("--tag", type=str, required=True, help="wandb tag")
def train(train_dir, val_dir, train_num_samples, val_num_samples, tag):
    wandb.init(project="computer-vision-papers", tags=[tag])

    train_dir = pathlib.Path(train_dir)
    val_dir = pathlib.Path(train_dir)

    BATCH_SIZE = 128
    CLASS_NAMES = np.array([item.name for item in train_dir.glob("*")])
    train_ds = build_dataset(train_dir, train_num_samples, BATCH_SIZE, CLASS_NAMES)
    val_ds = build_dataset(val_dir, val_num_samples, BATCH_SIZE, CLASS_NAMES)

    base_dir = ".\\models\\alexnet"
    last_model_checkpoint = get_last_model_checkpoint(base_dir)
    best_model_checkpoint = get_best_model_checkpoint(base_dir)
    last_model_manager = LastModelManager(base_dir)

    num_classes = CLASS_NAMES.shape[0]
    model = get_alexnet_model(num_classes)
    model, initial_epoch = get_latest_trained_model(model, base_dir)

    scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=10, min_lr=0.00001
    )

    model.fit(
        train_ds,
        epochs=90,
        initial_epoch=initial_epoch,
        steps_per_epoch=train_num_samples // BATCH_SIZE,
        validation_data=val_ds,
        validation_steps=val_num_samples // BATCH_SIZE,
        callbacks=[
            scheduler,
            last_model_checkpoint,
            best_model_checkpoint,
            last_model_manager,
            WandbCallback(save_model=False),
        ],
    )


if __name__ == "__main__":
    train()
