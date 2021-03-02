from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# Helper libraries


fashion_mnist = tf.keras.datasets.fashion_mnist


tf.enable_eager_execution()
