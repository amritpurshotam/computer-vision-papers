from tensorflow.keras import Sequential
from tensorflow.keras.layers import AveragePooling2D, Conv2D, Dense, Flatten
from tensorflow.keras.losses import categorical_crossentropy


class LeNet5(Sequential):
    """Implementation of the LeNet5 architecture.

    http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
    """

    def __init__(self):
        super().__init__()
        self.add(
            Conv2D(
                6,
                kernel_size=5,
                strides=1,
                activation="tanh",
                padding="same",
                input_shape=(28, 28, 1),
            )
        )
        self.add(AveragePooling2D(pool_size=(2, 2), strides=1, padding="valid"))
        self.add(
            Conv2D(16, kernel_size=5, strides=1, activation="tanh", padding="valid")
        )
        self.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding="valid"))
        self.add(Flatten())
        self.add(Dense(120, activation="tanh"))
        self.add(Dense(84, activation="tanh"))
        self.add(Dense(10, activation="softmax"))  # paper used euclidean RBF activation

        # need to check optimizer used in paper? SGD? learning rate?
        self.compile(
            optimizer="adam", loss=categorical_crossentropy, metrics=["accuracy"]
        )
