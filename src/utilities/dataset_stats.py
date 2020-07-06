import pathlib

import numpy as np
import tensorflow as tf

from src.utilities.image import (
    channel_mean,
    crop_center,
    resize_image_keep_aspect_ratio,
)


def process(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = resize_image_keep_aspect_ratio(image)
    image = crop_center(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


if __name__ == "__main__":
    batch_size = 1000
    num_images = 1281167
    data_dir = pathlib.Path("F:\\ILSVRC2012_img_train")
    dataset = tf.data.Dataset.list_files(str(data_dir / "*/*"))
    dataset = (
        dataset.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    means = np.zeros((num_images, 3))
    i = 0
    for batch in iter(dataset):
        batch_mean = tf.math.reduce_mean(batch, axis=(1, 2)).numpy()
        i_end = i + batch_size if i + batch_size <= num_images else num_images
        means[i:i_end] = batch_mean
        i = i + batch_size
    ds_means = np.mean(means, axis=0)

    print(ds_means)

    # [0.47900404 0.4560645  0.40068299]
    # [121.64567695 115.7963513  101.6740183 ]
