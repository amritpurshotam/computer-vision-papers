import tensorflow as tf

from src.utilities.math import cov

# https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py
_IMAGENET_PCA = {
    "eigval": tf.constant([0.2175, 0.0188, 0.0045], dtype=tf.float32),
    "eigvec": tf.constant(
        [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ],
        dtype=tf.float32,
    ),
}


def fancy_pca(
    img: tf.Tensor, imagenet_pca: bool = False, alpha_std: float = 0.1
) -> tf.Tensor:

    """PCA Colour Augmentation as described in AlexNet paper.

    Parameters
    ----------
    img : tf.Tensor
        3-dimensional Tensor of shape (h, w, 3)
    imagenet_pca : bool, optional
        Whether or not to use pre-computed imagenet principal components (from
        the whole dataset), by default False

    Returns
    -------
    tf.Tensor
        3-dimensional Tensor corresponding to the image with some noise added
        along the principal components of the colour channels.
    """
    rows, columns, _ = img.shape
    img = tf.reshape(img, (rows * columns, 3))

    mean = tf.reduce_mean(img, axis=0)
    std = tf.math.reduce_std(img, axis=0)
    img -= mean
    img /= std

    if imagenet_pca:
        lambdas = _IMAGENET_PCA["eigval"]
        p = _IMAGENET_PCA["eigvec"]
    else:
        covariance = cov(img, rowvar=False, bias=True)
        lambdas, p, _ = tf.linalg.svd(covariance)

    alphas = tf.random.normal((3,), 0, alpha_std)
    delta = tf.tensordot(p, alphas * lambdas, axes=1)

    img = img + delta
    img = img * std + mean
    img = tf.clip_by_value(img, 0, 255)

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
    return image


def crop_center_square(image):
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


def is_scaled(image: tf.Tensor) -> bool:
    """Checks if an image is scaled to between 0 and 1"""
    return tf.reduce_min(image) < 0 or tf.reduce_max(image) > 1


def subtract_mean(image: tf.Tensor) -> tf.Tensor:
    """Centers image pixel values per colour channel based on means from ImageNet.

    Parameters
    ----------
    image : tf.Tensor
        RGB image tensor with shape (h, w, 3) and pixel values already scaled to [0, 1]

    Returns
    -------
    tf.Tensor
        Centererd RGB image tensor

    Raises
    ------
    ValueError
        Raised if pixel values outside the range of [0,1]
    """
    if not is_scaled(image):
        raise ValueError("Image must have pixel values scaled to between [0,1]")

    mean = tf.constant([0.48105113, 0.45742367, 0.40778555], dtype=tf.float32)
    mean = tf.reshape(mean, [1, 1, 3])
    image = tf.math.subtract(image, mean)
    return image


def standardise(image: tf.Tensor) -> tf.Tensor:
    """Standardises image pixel values per colour channel based on standard deviation
    from ImageNet.

    Parameters
    ----------
    image : tf.Tensor
        RGB image tensor with shape (h, w, 3) and pixel values already scaled to [0, 1]

    Returns
    -------
    tf.Tensor
        Standardised RGB image tensor

    Raises
    ------
    ValueError
        Raised if pixel values outside the range of [0,1]
    """
    if not is_scaled(image):
        raise ValueError("Image must have pixel values scaled to between [0,1]")

    std = tf.constant([[0.2334365, 0.22940313, 0.23018445]], dtype=tf.float32)
    std = tf.reshape(std, [1, 1, 3])
    image = tf.math.divide(image, std)
    return image


def normalise(image: tf.Tensor) -> tf.Tensor:
    """Normalises image pixel values per colour channel based on means and
        standard deviation from ImageNet.

    Parameters
    ----------
    image : tf.Tensor
        RGB image tensor with shape (h, w, 3) and pixel values already scaled to [0, 1]

    Returns
    -------
    tf.Tensor
        Normalised RGB image tensor

    Raises
    ------
    ValueError
        Raised if pixel values outside the range of [0,1]
    """
    image = subtract_mean(image)
    image = standardise(image)
    return image
