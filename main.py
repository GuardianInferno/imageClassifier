import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#print(train_labels[0])
#print(train_images[0])

plt.imshow(train_images[0], cmap='gray', vmin=0, vmax=255)
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
])
