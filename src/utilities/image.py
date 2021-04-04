import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import Tensor

from src.utilities.math import cov_tf


def fancy_pca(original_img: Tensor):
    """PCA Colour Augmentation as described in AlexNet paper.

    Code adapted from https://github.com/ANONYMOUS-GURU/AlexNet/blob/master/Different%20layers/PCA_color_augmentation.py # noqa: B950
    """
    rows = original_img.shape[0]
    columns = original_img.shape[1]
    img = tf.reshape(original_img, (rows * columns, 3))

    img = tf.cast(img, "float32")
    mean = tf.reduce_mean(img, axis=0)
    std = tf.math.reduce_std(img, axis=0)
    img -= mean
    img /= std

    cov = cov_tf(img)
    lambdas, p, _ = tf.linalg.svd(cov)
    alphas = tf.random.normal((3,), 0, 0.1)
    delta = tf.tensordot(p, alphas * lambdas, axes=1)

    img = img + delta
    img = img * std + mean
    img = tf.clip_by_value(img, 0, 255)
    img = tf.cast(img, dtype=tf.uint8)

    img = tf.reshape(img, (rows, columns, 3))
    return img


def resize_image_keep_aspect_ratio(image, lo_dim=256):
    """Aspect ratio preserving image resize with the shorter dimension resized
    to equal lo_dim.

    Code adapted from https://stackoverflow.com/a/48648242 but converted to TF2.
    """
    initial_width = tf.shape(image)[0]
    initial_height = tf.shape(image)[1]

    min_dim = tf.math.minimum(initial_width, initial_height)
    ratio = tf.cast(min_dim, dtype=tf.float32) / tf.constant(lo_dim, dtype=tf.float32)

    new_width = tf.cast(
        tf.cast(initial_width, dtype=tf.float32) / ratio, dtype=tf.int32
    )
    new_height = tf.cast(
        tf.cast(initial_height, dtype=tf.float32) / ratio, dtype=tf.int32
    )

    image = tf.image.resize(image, [new_width, new_height])
    image = tf.cast(image, dtype=tf.uint8)
    return image


def crop_center(image):
    """Center crop largest square of image

    Code taken from https://stackoverflow.com/a/54866162
    """
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    if h > w:
        cropped = tf.image.crop_to_bounding_box(image, (h - w) // 2, 0, w, w)
    else:
        cropped = tf.image.crop_to_bounding_box(image, 0, (w - h) // 2, h, h)
    return cropped


def subtract_mean(image):
    mean = tf.constant([121.64567695, 115.7963513, 101.6740183], dtype=tf.float32)
    mean = tf.reshape(mean, [1, 1, 3])
    image = tf.math.subtract(image, mean)
    return image


if __name__ == "__main__":
    img = tf.io.read_file("cat.jpeg")
    img = tf.image.decode_jpeg(img)

    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5

    fig.add_subplot(rows, columns, 1)
    plt.imshow(img)
    for i in range(2, columns * rows + 1):
        pca = fancy_pca(img)
        fig.add_subplot(rows, columns, i)
        plt.imshow(pca)
    plt.show()
