import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPool2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD

from src.features.fancy_pca_tf import fancy_pca


class AlexNet(Sequential):
    def __init__(self):
        super().__init__()
        self.add(
            Conv2D(
                96,
                11,
                strides=4,
                padding="same",
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="zeros",
                input_shape=(224, 224, 3),
            )
        )
        self.add(
            Lambda(
                lambda X: tf.nn.lrn(X, bias=2, depth_radius=5, alpha=0.0001, beta=0.75)
            )
        )
        self.add(MaxPool2D(pool_size=3, strides=2, padding="valid"))

        self.add(
            Conv2D(
                256,
                5,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="ones",
            )
        )
        self.add(
            Lambda(
                lambda X: tf.nn.lrn(X, bias=2, depth_radius=5, alpha=0.0001, beta=0.75)
            )
        )
        self.add(MaxPool2D(pool_size=3, strides=2, padding="valid"))

        self.add(
            Conv2D(
                384,
                3,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="zeros",
            )
        )
        self.add(
            Conv2D(
                384,
                3,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="ones",
            )
        )
        self.add(
            Conv2D(
                256,
                3,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="ones",
            )
        )
        self.add(MaxPool2D(pool_size=3, strides=2, padding="valid"))

        self.add(Flatten())
        self.add(
            Dense(
                4096,
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="ones",
            )
        )
        self.add(Dropout(0.5))
        self.add(
            Dense(
                4096,
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="ones",
            )
        )
        self.add(Dropout(0.5))
        self.add(
            Dense(
                1000,
                activation="softmax",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="zeros",
            )
        )

        optimizer = SGD(learning_rate=0.01, momentum=0.9, decay=0.0005)
        self.compile(
            optimizer=optimizer, loss=categorical_crossentropy, metrics=["accuracy"]
        )

        # data augmentation: alter intensities of RGB. See paper for details


def resize_image_keep_aspect_ratio(image, lo_dim=256):
    """Aspect ratio preserving image resize with the shorter dimension resized to equal lo_dim.

    Code inspired from https://stackoverflow.com/a/48648242 but converted to TF2.
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
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    if h > w:
        cropped = tf.image.crop_to_bounding_box(image, (h - w) // 2, 0, w, w)
    else:
        cropped = tf.image.crop_to_bounding_box(image, 0, (w - h) // 2, h, h)
    return cropped


def process_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = resize_image_keep_aspect_ratio(img)
    img = crop_center(img)
    img = tf.image.random_crop(img, size=[224, 224, 3])
    img = tf.image.random_flip_left_right(img)
    img = fancy_pca(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def show_batch(image_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    model = AlexNet()
    scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=10, min_lr=0.00001
    )
    # model.fit([0], [0], batch_size=128, epochs=90, callbacks=[scheduler])

    model.summary()

    data_dir = "F:/Test/*/*"
    dataset = (
        tf.data.Dataset.list_files(data_dir)
        .repeat()
        .shuffle(250)
        .map(process_image)
        .batch(25)
        .prefetch(1)
    )

    batch = dataset.take(25)
    print(type(batch))
    show_batch(list(batch))
