from tensorflow.keras import Input, Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPool2D,
)
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import L2

from src.layers.local_response_normalisation import LocalResponseNormalization
from src.models.alexnet.hyperparams import (
    CHANNELS,
    IMAGE_SIZE,
    KERNEL_MEAN,
    KERNEL_STDDEV,
    L2_PENALTY,
    LR,
    MOMENTUM,
)


def get_alexnet_model(num_class: int) -> Sequential:
    image = Input((IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype="float32", name="image")

    c1_a = Conv2D(
        filters=48,
        kernel_size=11,
        strides=4,
        padding="valid",
        activation="relu",
        kernel_initializer=RandomNormal(mean=KERNEL_MEAN, stddev=KERNEL_STDDEV),
        bias_initializer="zeros",
        kernel_regularizer=L2(l2=L2_PENALTY),
        name="C1A",
    )(image)
    lrn1_a = LocalResponseNormalization(name="LRN1A")(c1_a)
    mp1_a = MaxPool2D(pool_size=3, strides=2, padding="valid", name="MP1A")(lrn1_a)

    c2_a = Conv2D(
        filters=128,
        kernel_size=5,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=RandomNormal(mean=KERNEL_MEAN, stddev=KERNEL_STDDEV),
        bias_initializer="ones",
        kernel_regularizer=L2(l2=L2_PENALTY),
        name="C2A",
    )(mp1_a)
    lrn2_a = LocalResponseNormalization(name="LRN2A")(c2_a)
    mp2_a = MaxPool2D(pool_size=3, strides=2, padding="valid", name="MP2A")(lrn2_a)

    c1_b = Conv2D(
        filters=48,
        kernel_size=11,
        strides=4,
        padding="valid",
        activation="relu",
        kernel_initializer=RandomNormal(mean=KERNEL_MEAN, stddev=KERNEL_STDDEV),
        bias_initializer="zeros",
        kernel_regularizer=L2(l2=L2_PENALTY),
        name="C1B",
    )(image)
    lrn1_bottom = LocalResponseNormalization(name="LRN1B")(c1_b)
    mp1_b = MaxPool2D(pool_size=3, strides=2, padding="valid", name="MP1B")(lrn1_bottom)

    c2_b = Conv2D(
        filters=128,
        kernel_size=5,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=RandomNormal(mean=KERNEL_MEAN, stddev=KERNEL_STDDEV),
        bias_initializer="ones",
        kernel_regularizer=L2(l2=L2_PENALTY),
        name="C2B",
    )(mp1_b)
    lrn2_b = LocalResponseNormalization(name="LRN2B")(c2_b)
    mp2_b = MaxPool2D(pool_size=3, strides=2, padding="valid", name="MP2B")(lrn2_b)

    concat_1 = Concatenate(name="Concat2")([mp2_a, mp2_b])

    c3_a = Conv2D(
        192,
        3,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=RandomNormal(mean=KERNEL_MEAN, stddev=KERNEL_STDDEV),
        bias_initializer="zeros",
        kernel_regularizer=L2(l2=L2_PENALTY),
        name="C3A",
    )(concat_1)
    c4_a = Conv2D(
        192,
        3,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=RandomNormal(mean=KERNEL_MEAN, stddev=KERNEL_STDDEV),
        bias_initializer="ones",
        kernel_regularizer=L2(l2=L2_PENALTY),
        name="C4A",
    )(c3_a)
    c5_a = Conv2D(
        128,
        3,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=RandomNormal(mean=KERNEL_MEAN, stddev=KERNEL_STDDEV),
        bias_initializer="ones",
        kernel_regularizer=L2(l2=L2_PENALTY),
        name="C5A",
    )(c4_a)
    mp5_a = MaxPool2D(pool_size=3, strides=2, padding="valid", name="MP5A")(c5_a)
    flat5_a = Flatten(name="F5A")(mp5_a)

    c3_b = Conv2D(
        192,
        3,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=RandomNormal(mean=KERNEL_MEAN, stddev=KERNEL_STDDEV),
        bias_initializer="zeros",
        kernel_regularizer=L2(l2=L2_PENALTY),
        name="C3B",
    )(concat_1)
    c4_b = Conv2D(
        192,
        3,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=RandomNormal(mean=KERNEL_MEAN, stddev=KERNEL_STDDEV),
        bias_initializer="ones",
        kernel_regularizer=L2(l2=L2_PENALTY),
        name="C4B",
    )(c3_b)
    c5_b = Conv2D(
        128,
        3,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=RandomNormal(mean=KERNEL_MEAN, stddev=KERNEL_STDDEV),
        bias_initializer="ones",
        kernel_regularizer=L2(l2=L2_PENALTY),
        name="C5B",
    )(c4_b)
    mp5_bottom = MaxPool2D(pool_size=3, strides=2, padding="valid", name="MP5B")(c5_b)
    flat5_b = Flatten(name="F5B")(mp5_bottom)

    concat_2 = Concatenate(name="Concat5")([flat5_a, flat5_b])

    fc6 = Dense(
        4096,
        activation="relu",
        kernel_initializer=RandomNormal(mean=KERNEL_MEAN, stddev=KERNEL_STDDEV),
        bias_initializer="ones",
        kernel_regularizer=L2(l2=L2_PENALTY),
        name="FC6",
    )(concat_2)
    dropout6 = Dropout(rate=0.5, name="DO6")(fc6)
    fc7 = Dense(
        4096,
        activation="relu",
        kernel_initializer=RandomNormal(mean=KERNEL_MEAN, stddev=KERNEL_STDDEV),
        bias_initializer="ones",
        kernel_regularizer=L2(l2=L2_PENALTY),
        name="FC7",
    )(dropout6)
    dropout7 = Dropout(rate=0.5, name="DO7")(fc7)
    classifier = Dense(
        num_class,
        activation="softmax",
        kernel_initializer=RandomNormal(mean=KERNEL_MEAN, stddev=KERNEL_STDDEV),
        bias_initializer="zeros",
        kernel_regularizer=L2(l2=L2_PENALTY),
        name="Classifier",
    )(dropout7)

    model = Model(image, classifier)

    optimizer = SGD(learning_rate=LR, momentum=MOMENTUM)
    model.compile(
        optimizer=optimizer,
        loss=categorical_crossentropy,
        metrics=["accuracy", TopKCategoricalAccuracy(k=5)],
    )

    return model
