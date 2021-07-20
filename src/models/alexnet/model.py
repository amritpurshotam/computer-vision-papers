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


def get_alexnet_model(num_class: int) -> Sequential:
    image_input = Input((227, 227, 3), dtype="float32", name="image")

    c1_top = Conv2D(
        filters=48,
        kernel_size=11,
        strides=4,
        padding="valid",
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer="zeros",
        kernel_regularizer=L2(l2=0.0005),
    )(image_input)
    lrn1_top = LocalResponseNormalization()(c1_top)
    mp1_top = MaxPool2D(pool_size=3, strides=2, padding="valid")(lrn1_top)

    c2_top = Conv2D(
        filters=128,
        kernel_size=5,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer="ones",
        kernel_regularizer=L2(l2=0.0005),
    )(mp1_top)
    lrn2_top = LocalResponseNormalization()(c2_top)
    mp2_top = MaxPool2D(pool_size=3, strides=2, padding="valid")(lrn2_top)

    c1_bottom = Conv2D(
        filters=48,
        kernel_size=11,
        strides=4,
        padding="valid",
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer="zeros",
        kernel_regularizer=L2(l2=0.0005),
    )(image_input)
    lrn1_bottom = LocalResponseNormalization()(c1_bottom)
    mp1_bottom = MaxPool2D(pool_size=3, strides=2, padding="valid")(lrn1_bottom)

    c2_bottom = Conv2D(
        filters=128,
        kernel_size=5,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer="ones",
        kernel_regularizer=L2(l2=0.0005),
    )(mp1_bottom)
    lrn2_bottom = LocalResponseNormalization()(c2_bottom)
    mp2_bottom = MaxPool2D(pool_size=3, strides=2, padding="valid")(lrn2_bottom)

    concat_1 = Concatenate()([mp2_top, mp2_bottom])

    c3_top = Conv2D(
        192,
        3,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer="zeros",
        kernel_regularizer=L2(l2=0.0005),
    )(concat_1)
    c4_top = Conv2D(
        192,
        3,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer="ones",
        kernel_regularizer=L2(l2=0.0005),
    )(c3_top)
    c5_top = Conv2D(
        128,
        3,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer="ones",
        kernel_regularizer=L2(l2=0.0005),
    )(c4_top)
    mp5_top = MaxPool2D(pool_size=3, strides=2, padding="valid")(c5_top)
    flat5_top = Flatten()(mp5_top)

    c3_bottom = Conv2D(
        192,
        3,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer="zeros",
        kernel_regularizer=L2(l2=0.0005),
    )(concat_1)
    c4_bottom = Conv2D(
        192,
        3,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer="ones",
        kernel_regularizer=L2(l2=0.0005),
    )(c3_bottom)
    c5_bottom = Conv2D(
        128,
        3,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer="ones",
        kernel_regularizer=L2(l2=0.0005),
    )(c4_bottom)
    mp5_bottom = MaxPool2D(pool_size=3, strides=2, padding="valid")(c5_bottom)
    flat5_bottom = Flatten()(mp5_bottom)

    concat_2 = Concatenate()([flat5_top, flat5_bottom])

    dense6 = Dense(
        4096,
        activation="relu",
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer="ones",
        kernel_regularizer=L2(l2=0.0005),
    )(concat_2)
    dropout6 = Dropout(0.5)(dense6)
    dense7 = Dense(
        4096,
        activation="relu",
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer="ones",
        kernel_regularizer=L2(l2=0.0005),
    )(dropout6)
    dropout7 = Dropout(0.5)(dense7)
    classifier = Dense(
        num_class,
        activation="softmax",
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer="zeros",
        kernel_regularizer=L2(l2=0.0005),
    )(dropout7)

    model = Model(image_input, classifier)

    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(
        optimizer=optimizer,
        loss=categorical_crossentropy,
        metrics=["accuracy", TopKCategoricalAccuracy(k=5)],
    )

    return model
