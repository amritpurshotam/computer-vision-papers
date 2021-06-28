import tensorflow as tf


def cov(m: tf.Tensor, rowvar: bool = False) -> tf.Tensor:
    """Estimate a covariance matrix.
    Mimics the behaviour of `np.cov(m)` with default parameters i.e. `bias=False`
    and `rowvar=True`.

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
        normalization_factor = tf.cast(tf.shape(m)[1] - 1, tf.float32)
        covariance = tf.matmul(m, tf.transpose(m)) / normalization_factor
        return covariance
    else:
        m = m - tf.reduce_mean(m, axis=0, keepdims=True)
        normalization_factor = tf.cast(tf.shape(m)[0] - 1, tf.float32)
        covariance = tf.matmul(tf.transpose(m), m) / normalization_factor
        return covariance
