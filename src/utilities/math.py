import tensorflow as tf


def cov(m: tf.Tensor, rowvar: bool = True, bias: bool = False) -> tf.Tensor:
    """Estimate a covariance matrix.
    Mimics the behaviour of `np.cov(m)`

    Parameters
    ----------
    m : tf.Tensor
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of m represents a variable, and each column a single
        observation of all those variables.

    rowvar: bool
        If rowvar is True (default), then each row represents a variable, with
        observations in the columns. Otherwise, the relationship is transposed:
        each column represents a variable, while the rows contain observations.

    Returns
    -------
    tf.Tensor
        The covariance matrix of the variables.
    """
    if rowvar:
        m = m - tf.reduce_mean(m, axis=1, keepdims=True)
        n = tf.shape(m)[1] if bias else tf.shape(m)[1] - 1
        covariance = tf.matmul(m, tf.transpose(m)) / tf.cast(n, tf.float32)
        return covariance
    else:
        m = m - tf.reduce_mean(m, axis=0, keepdims=True)
        n = tf.shape(m)[0] if bias else tf.shape(m)[0] - 1
        covariance = tf.matmul(tf.transpose(m), m) / tf.cast(n, tf.float32)
        return covariance
