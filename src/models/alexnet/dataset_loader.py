import os
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from src.models.alexnet.hyperparams import BATCH_SIZE, CHANNELS, IMAGE_SIZE
from src.utilities.image import (
    crop_center_square,
    fancy_pca,
    resize_image_keep_aspect_ratio,
    subtract_mean,
)


class DatasetLoader:
    def __init__(
        self,
        train_dir: Path,
        train_samples: int,
        val_dir: Path,
        val_samples: int,
        apply_pca: bool,
        imagenet_pca: bool,
    ) -> None:
        self.train_dir = train_dir
        self.train_samples = train_samples
        self.val_dir = val_dir
        self.val_samples = val_samples
        self.apply_pca = apply_pca
        self.imagenet_pca = imagenet_pca
        self.batch_size = BATCH_SIZE
        self.class_names = np.array([item.name for item in self.train_dir.glob("*")])

    @property
    def num_classes(self):
        return self.class_names.shape[0]

    def __get_label(self, file_path: str):
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2] == self.class_names

    def __decode_image(self, img_path: str) -> tf.Tensor:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=CHANNELS)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def __resize_and_crop(self, img: tf.Tensor) -> tf.Tensor:
        img = resize_image_keep_aspect_ratio(img)
        img = crop_center_square(img)
        return img

    def __augment(self, img: tf.Tensor) -> tf.Tensor:
        img = tf.image.random_crop(img, size=[IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
        img = tf.image.random_flip_left_right(img)
        if self.apply_pca:
            img = fancy_pca(img, self.imagenet_pca)
        return img

    def __process_path(
        self,
        file_path: str,
        training: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        label = self.__get_label(file_path)
        image = self.__decode_image(file_path)
        image = self.__resize_and_crop(image)
        if training:
            image = self.__augment(image)
        image = subtract_mean(image)
        return image, label

    def __build_dataset(
        self,
        data_dir: Path,
        num_samples: int,
        training: bool,
    ) -> tf.data.Dataset:
        ds = (
            tf.data.Dataset.list_files(str(data_dir / "*/*"))
            .shuffle(num_samples)
            .map(
                lambda file_path: self.__process_path(
                    file_path,
                    training,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(self.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        return ds

    def build_train_ds(self) -> tf.data.Dataset:
        return self.__build_dataset(self.train_dir, self.train_samples, True)

    def build_val_ds(self) -> tf.data.Dataset:
        return self.__build_dataset(self.val_dir, self.val_samples, False)
