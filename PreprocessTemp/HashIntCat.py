# Given is the code for applying hashing trick for interger categorical features

import tensorflow as tf
import numpy as np

IntData = np.random.randint(0, 100000, size = (10000, 1))

Hasher = tf.keras.layers.Hashing(num_bins = 64, salt = 1337)
Encoder = tf.keras.layers.CategoryEncoding(num_tokens = 64, output_mode = "multi_hot")
EncodedData = Encoder(Hasher(IntData))
print(EncodedData.shape)
