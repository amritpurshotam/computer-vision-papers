from pathlib import Path
from typing import Tuple

import click
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import get_dataset_config
from src.utilities.image import crop_center, resize_image_keep_aspect_ratio


def load_single_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path, formats=["JPEG"])
    if image.mode == "CMYK" or image.mode == "L":
        image = image.convert("RGB")
    elif image.mode != "RGB":
        print(f"{image.mode}: {image_path}")
    image = np.asarray(image)
    image = image.astype(np.float32)
    image = image / 255.0
    return image


def calculate_stats_from_full_images(
    train_dir: Path, num_images: int
) -> Tuple[np.ndarray, np.ndarray]:
    image_paths = train_dir.glob("*/*")
    means = np.zeros((num_images, 3))
    stds = np.zeros((num_images, 3))
    i = 0
    for image_path in iter(image_paths):
        try:
            image = load_single_image(image_path)
            mean = np.mean(image, axis=(0, 1))
            std = np.std(image, axis=(0, 1))
            means[i] = mean
            stds[i] = std
            i = i + 1
        except Exception as e:
            print(image_path)
            print(e)
    ds_means = np.mean(means, axis=0)
    ds_stds = np.mean(stds, axis=0)

    return ds_means, ds_stds


def load_image_batch(image_path: str):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = resize_image_keep_aspect_ratio(image, lo_dim=256)
    image = crop_center(image)  # shape of (256, 256, 3)
    image = tf.image.central_crop(image, 0.875)  # shape of (224, 224, 3)
    return image


def calculate_stats_from_cropped_images(
    train_dir: Path, num_images: int
) -> Tuple[np.ndarray, np.ndarray]:
    batch_size = 1000
    ds = (
        tf.data.Dataset.list_files(str(train_dir / "*/*"))
        .map(
            load_image_batch,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    means = np.zeros((num_images, 3))
    stds = np.zeros((num_images, 3))

    i = 0
    for batch in iter(ds):
        batch_mean = tf.math.reduce_mean(batch, axis=(1, 2)).numpy()
        batch_std = tf.math.reduce_std(batch, axis=(1, 2)).numpy()
        print(batch_mean.shape)

        i_end = i + batch_size if i + batch_size <= num_images else num_images
        means[i:i_end] = batch_mean
        stds[i:i_end] = batch_std
        i = i + batch_size

    ds_means = np.mean(means, axis=0)
    ds_stds = np.mean(stds, axis=0)

    return ds_means, ds_stds


@click.command()
@click.option(
    "--image_type", type=click.Choice(["full", "cropped"], case_sensitive=False)
)
def main(image_type: str):
    ds_config = get_dataset_config("imagenet")
    train_dir = ds_config["train"]["path"]
    num_images = ds_config["train"]["samples"]

    if image_type == "full":
        ds_means, ds_stds = calculate_stats_from_full_images(train_dir, num_images)
    elif image_type == "cropped":
        ds_means, ds_stds = calculate_stats_from_cropped_images(train_dir, num_images)

    print(f"Per channel means: {ds_means}")
    print(f"Per channel standard deviations: {ds_stds}")


if __name__ == "__main__":
    main()
