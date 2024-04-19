from typing import Tuple

import keras_tuner as kt
import tensorflow as tf


class Model1(kt.HyperModel):
    def __init__(self, num_classes: int, input_shape: Tuple[int, int], desired_shape: Tuple[int, int, int]):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.desired_shape = desired_shape
        super(Model1)

    def build(self, hp):
        try:
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Input(self.input_shape))
            model.add(tf.keras.layers.Reshape(self.desired_shape))
            model.add(tf.keras.layers.Conv2D(
                #adding filter
                filters=hp.Choice('conv_2_filter', values=[20, 30]),
                # adding filter size or kernel size
                kernel_size=hp.Choice('conv_2_kernel', values=[2, 4]),
                #activation function
                activation=hp.Choice('activation', ['relu'])))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=hp.Choice('pool_size', [2, 3]), strides=hp.Choice('strides', [2, 3]), padding='valid'))
            model.add(tf.keras.layers.Dropout(rate=hp.Choice("dropout", [0.1, 0.25, 0.5])))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(self.num_classes, activation="softmax"))

            # compilation of model
            model.compile(optimizer=hp.Choice('optimizer', values=[
                              'rmsprop',
                              'adam',
                          ]),
                          loss=hp.Choice('loss', values=[
                              'sparse_categorical_crossentropy',
                              'categorical_crossentropy',
                          ]),
                          metrics=['accuracy'])

            model.summary()

            return model
        except Exception as e:
            print(f'An Exception occured while building the model: {e}')
            return None

    def fit(self, hp, _model, *args, **kwargs):
        return _model.fit(
            *args,
            batch_size=hp.Choice("batch_size", values=[20, 100]),
            shuffle=True,
            **kwargs,
        )