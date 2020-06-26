import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import Tensor


def cov_tf(img):
    """Covariance matrix calculation

    Code taken from https://stackoverflow.com/a/49850652
    """
    mean_x = tf.reduce_mean(img, axis=0, keepdims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(img), img) / tf.cast(tf.shape(img)[0], tf.float32)
    cov = vx - mx
    return cov


def fancy_pca(original_img: Tensor):
    """PCA Colour Augmentation as described in AlexNet paper.

    Code adapted from https://github.com/ANONYMOUS-GURU/AlexNet/blob/master/Different%20layers/PCA_color_augmentation.py
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
