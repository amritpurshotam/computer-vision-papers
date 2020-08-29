import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPool2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD


class AlexNet(Sequential):
    def __init__(self, num_class: int):
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
                num_class,
                activation="softmax",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="zeros",
            )
        )

        optimizer = SGD(learning_rate=0.01, momentum=0.9, decay=0.0005)
        self.compile(
            optimizer=optimizer, loss=categorical_crossentropy, metrics=["accuracy"]
        )
