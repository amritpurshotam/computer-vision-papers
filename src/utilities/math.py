import tensorflow as tf


def cov_tf(img):
    """Covariance matrix calculation

    Code taken from https://stackoverflow.com/a/49850652
    """
    mean_x = tf.reduce_mean(img, axis=0, keepdims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(img), img) / tf.cast(tf.shape(img)[0], tf.float32)
    cov = vx - mx
    return cov
