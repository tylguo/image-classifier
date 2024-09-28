import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from PIL import Image

import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()