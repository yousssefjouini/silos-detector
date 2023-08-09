from keras.models import load_model
import importlib.resources
import os
from keras.utils import load_img, img_to_array
from typing import Any
from PIL import Image
import numpy as np
from keras.layers import Layer
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import UpSampling2D
from keras.layers import concatenate
from keras.layers import Add
from keras.layers import Multiply
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization
import tensorflow as tf


class EncoderBlock(Layer):
    def __init__(self, filters, rate, pooling=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.filters = filters
        self.rate = rate
        self.pooling = pooling
        self.c1 = Conv2D(
            filters,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
        )
        self.drop = Dropout(rate)
        self.c2 = Conv2D(
            filters,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
        )
        self.pool = MaxPool2D()

    def call(self, X):
        x = self.c1(X)
        x = self.drop(x)
        x = self.c2(x)
        if self.pooling:
            y = self.pool(x)
            return y, x
        else:
            return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            "rate": self.rate,
            "pooling": self.pooling,
        }


class DecoderBlock(Layer):
    def __init__(self, filters, rate, **kwargs) -> None:
        super(DecoderBlock, self).__init__(**kwargs)
        self.filters = filters
        self.rate = rate
        self.up = UpSampling2D()
        self.net = EncoderBlock(filters, rate, pooling=False)

    def call(self, X):
        X, skip_X = X
        x = self.up(X)
        c_ = concatenate([x, skip_X])
        x = self.net(c_)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            "rate": self.rate,
        }


class AttentionGate(Layer):
    def __init__(self, filters, bn, **kwargs) -> None:
        super(AttentionGate, self).__init__(**kwargs)
        self.filters = filters
        self.bn = bn
        self.normal = Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
        )
        self.down = Conv2D(
            filters,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
        )
        self.learn = Conv2D(1, kernel_size=1, padding="same", activation="sigmoid")
        self.resample = UpSampling2D()
        self.BN = BatchNormalization()

    def call(self, X):
        X, skip_X = X
        x = self.normal(X)
        skip = self.down(skip_X)
        x = Add()([x, skip])
        x = self.learn(x)
        x = self.resample(x)
        f = Multiply()([x, skip_X])
        if self.bn:
            return self.BN(f)
        else:
            return f

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "filters": self.filters, "bn": self.bn}


def dice_coef(y_true, y_pred, threshold=0.8):
    y_pred_f = tf.cast(y_pred > threshold, tf.float32)
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred_f, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 0.00001) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 0.00001
    )


class SilosSegmentation:
    """Loads a CNN segmentation model to predict on files.
    predict(filepath) : given a filepath, returns a np.array corresponding to the predicted mask
    """

    def __init__(self):
        with importlib.resources.path(
            "silos.segmentation.prediction.models", "segment1.h5"
        ) as model_path:
            self.model = load_model(
                model_path,
                custom_objects={
                    "EncoderBlock": EncoderBlock,
                    "DecoderBlock": DecoderBlock,
                    "AttentionGate": AttentionGate,
                    "dice_coef": dice_coef,
                },
                compile=False,
            )

    def predict(self, to_predict: str, from_buffer: bool = False) -> Any:
        if from_buffer:
            uploaded_file = Image.open(to_predict)
            uploaded_file = np.array(uploaded_file)
            uploaded_file = uploaded_file / 255
            uploaded_file = uploaded_file[None, :]
            prediction = self.model.predict(uploaded_file)
            prediction = tf.cast(prediction > 0.4, tf.float32)
            return prediction

        elif os.path.isfile(to_predict):
            img = load_img(to_predict, target_size=(32, 32, 1), grayscale=False)
            img = img_to_array(img)
            img = img / 255
            img = img[None, :]
            prediction = self.model.predict(img)
            return prediction
