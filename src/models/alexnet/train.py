import os
from pathlib import Path

import click
import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from wandb.keras import WandbCallback

from src.callbacks.helper import get_best_model_checkpoint
from src.config import get_dataset_config
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
    img = tf.cast(img, tf.float32)
    return img


def resize_and_crop(img: tf.Tensor) -> tf.Tensor:
    img = resize_image_keep_aspect_ratio(img)
    img = crop_center(img)
    return img


def augment(img: tf.Tensor, apply_pca: bool, imagenet_pca: bool):
    img = tf.image.random_crop(img, size=[227, 227, 3])
    img = tf.image.random_flip_left_right(img)
    if apply_pca:
        img = fancy_pca(img, imagenet_pca)
    return img


def process_path(
    file_path: str,
    class_names: np.ndarray,
    training: bool,
    apply_pca: bool,
    imagenet_pca: bool,
):
    label = get_label(file_path, class_names)
    image = decode_image(file_path)
    image = resize_and_crop(image)
    if training:
        image = augment(image, apply_pca, imagenet_pca)
    image = subtract_mean(image)
    return image, label


def build_dataset(
    data_dir: Path,
    num_samples: int,
    batch_size: int,
    class_names: np.ndarray,
    training: bool,
    apply_pca: bool,
    imagenet_pca: bool,
):
    #  repeat -> shuffle -> map -> batch -> batch-wise map -> prefetch
    ds = (
        tf.data.Dataset.list_files(str(data_dir / "*/*"))
        .shuffle(num_samples)
        .map(
            lambda file_path: process_path(
                file_path, class_names, training, apply_pca, imagenet_pca
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return ds


@click.command()
@click.option(
    "--dataset", type=click.Choice(["imagenet", "imagenette"], case_sensitive=False)
)
@click.option(
    "--group", type=str, required=True, help="What to organise this experiment under"
)
@click.option(
    "--name", type=str, required=True, help="The name of this specific experiment"
)
@click.option(
    "--apply_pca",
    type=bool,
    required=False,
    default=True,
    help="Whether or not to apply PCA Augmentation",
)
@click.option(
    "--imagenet_pca",
    type=bool,
    required=False,
    default=False,
    help=(
        "Whether or not to use pre-computed principal components from the "
        + "whole of ImageNet"
    ),
)
def train(dataset: str, group: str, name: str, apply_pca: bool, imagenet_pca: bool):
    wandb.init(project="computer-vision-papers", group=group, name=name, tags=[dataset])

    ds_config = get_dataset_config(dataset)
    train_dir = ds_config["train"]["path"]
    val_dir = ds_config["val"]["path"]
    train_samples = ds_config["train"]["samples"]
    val_samples = ds_config["val"]["samples"]

    BATCH_SIZE = 128
    CLASS_NAMES = np.array([item.name for item in train_dir.glob("*")])
    train_ds = build_dataset(
        train_dir, train_samples, BATCH_SIZE, CLASS_NAMES, True, apply_pca, imagenet_pca
    )
    val_ds = build_dataset(
        val_dir, val_samples, BATCH_SIZE, CLASS_NAMES, False, apply_pca, imagenet_pca
    )

    base_dir = f".\\models\\alexnet\\{dataset}\\{group}\\{name}"
    best_model_checkpoint = get_best_model_checkpoint(base_dir)

    num_classes = CLASS_NAMES.shape[0]
    model = get_alexnet_model(num_classes)

    scheduler = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.1,
        mode="max",
        patience=8,
        min_lr=0.00001,
        min_delta=0.0001,
    )

    model.fit(
        train_ds,
        epochs=90,
        validation_data=val_ds,
        callbacks=[
            scheduler,
            best_model_checkpoint,
            WandbCallback(
                save_model=False,
                log_weights=True,
                log_gradients=True,
                training_data=train_ds,
            ),
        ],
    )

    model.save(f"{base_dir}_model")


if __name__ == "__main__":
    train()
