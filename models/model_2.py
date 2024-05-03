from typing import Tuple
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.regularizers import l1_l2

class Model2(kt.HyperModel):
    def __init__(self, num_classes: int, input_shape: Tuple[int, int], desired_shape: Tuple[int, int, int]):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.desired_shape = desired_shape
        super(Model2, self).__init__()

    def build(self, hp):
        try:
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Input(shape=self.input_shape))
            # Data augmentation layers
            # Rotates image by ([factor] * 360) degrees in either direction
            # Model is extremely sensitive to rotation and larger values confuse it
            model.add(tf.keras.layers.RandomRotation(factor=hp.Choice('factor', [0.001, 0.0001])))
            model.add(tf.keras.layers.Reshape(self.desired_shape))
            # Conv2D layer with L1/L2 regularization
            model.add(tf.keras.layers.Conv2D(
                filters=hp.Choice('conv_2_filter', values=[20, 30]),
                kernel_size=hp.Choice('conv_2_kernel', values=[2, 4]),
                activation=hp.Choice('activation', ['relu']),
                kernel_regularizer=l1_l2(l1=hp.Choice('l1', [0.01, 0.1]), l2=hp.Choice('l2', [0.01, 0.1]))
            ))
            model.add(tf.keras.layers.MaxPooling2D(
                pool_size=hp.Choice('pool_size', [2, 3]),
                strides=hp.Choice('strides', [2, 3]),
                padding='valid'
            ))
            model.add(tf.keras.layers.Dropout(rate=hp.Choice("dropout", [0.1, 0.25, 0.5])))
            # Conv2D layer with L1/L2 regularization
            model.add(tf.keras.layers.Conv2D(
                filters=hp.Choice('conv_2_filter', values=[60, 120]),
                kernel_size=hp.Choice('conv_2_kernel', values=[2, 4]),
                activation=hp.Choice('activation', ['relu']),
                kernel_regularizer=l1_l2(l1=hp.Choice('l1', [0.01, 0.1]), l2=hp.Choice('l2', [0.01, 0.1]))
            ))
            model.add(tf.keras.layers.MaxPooling2D(
                pool_size=hp.Choice('pool_size', [2, 3]),
                strides=hp.Choice('strides', [2, 3]),
                padding='valid'
            ))
            model.add(tf.keras.layers.Dropout(rate=hp.Choice("dropout", [0.1, 0.25, 0.5])))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(self.num_classes, activation="softmax"))

            # Compilation of the model
            model.compile(
                optimizer=hp.Choice('optimizer', values=['rmsprop', 'adam']),
                loss=hp.Choice('loss', values=[
                    'sparse_categorical_crossentropy',
                    'categorical_crossentropy',
                ]),
                metrics=['accuracy'])

            model.summary()

            return model
        except Exception as e:
            print(f'An Exception occurred while building the model: {e}')
            return None

    def fit(self, hp, _model, *args, **kwargs):
        return _model.fit(
            *args,
            batch_size=hp.Choice("batch_size", values=[20, 100]),
            shuffle=True,
            **kwargs,
        )
