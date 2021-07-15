from tensorflow.keras import Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import L2

from src.layers.local_response_normalisation import LocalResponseNormalization


def get_alexnet_model(num_class: int) -> Sequential:
    model = Sequential(
        [
            Conv2D(
                96,
                11,
                strides=4,
                padding="same",
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="zeros",
                kernel_regularizer=L2(l2=0.0005),
                input_shape=(224, 224, 3),
            ),
            LocalResponseNormalization(),
            MaxPool2D(pool_size=3, strides=2, padding="valid"),
            Conv2D(
                256,
                5,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="ones",
                kernel_regularizer=L2(l2=0.0005),
            ),
            LocalResponseNormalization(),
            MaxPool2D(pool_size=3, strides=2, padding="valid"),
            Conv2D(
                384,
                3,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="zeros",
                kernel_regularizer=L2(l2=0.0005),
            ),
            Conv2D(
                384,
                3,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="ones",
                kernel_regularizer=L2(l2=0.0005),
            ),
            Conv2D(
                256,
                3,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="ones",
                kernel_regularizer=L2(l2=0.0005),
            ),
            MaxPool2D(pool_size=3, strides=2, padding="valid"),
            Flatten(),
            Dense(
                4096,
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="ones",
                kernel_regularizer=L2(l2=0.0005),
            ),
            Dropout(0.5),
            Dense(
                4096,
                activation="relu",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="ones",
                kernel_regularizer=L2(l2=0.0005),
            ),
            Dropout(0.5),
            Dense(
                num_class,
                activation="softmax",
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                bias_initializer="zeros",
                kernel_regularizer=L2(l2=0.0005),
            ),
        ]
    )

    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(
        optimizer=optimizer,
        loss=categorical_crossentropy,
        metrics=["accuracy", TopKCategoricalAccuracy(k=5)],
    )

    return model
