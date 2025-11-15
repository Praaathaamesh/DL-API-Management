# Given is the template code for Image Augmentation using Keras

import tensorflow as tf
import numpy as np

# Scheme Sequential
DataAugTemp = tf.keras.Sequential(
        [tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)]
        )

# Load/map the data
(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
input_shape = x_train.shape[1:]
classes = 10

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.batch(16).map(lambda x, y: (DataAugTemp(x), y))

# Instantiate model architecture
Input = tf.keras.Input(shape = input_shape)
middle = tf.keras.layers.Rescaling(1.0 / 255)(Input) # Here we are rescaling the inputs
Output = tf.keras.applications.ResNet50(
        weights = None, input_shape = input_shape, classes = classes
        )(middle)

# Create/train model

Model = tf.keras.Model(Input, Output)
Model.compile(optimizer = "rmsprop", loss = "sparse_categorical_crossentropy")
Model.fit(train_data, steps_per_epoch = 5)
