import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import layers, models

# Zero-dimensional
x = np.array(22)
print(x.ndim)

# One-dimensional
x = np.array([4, 28, 16, 5, 8])
print(x.ndim)

# Two-dimensional
x = np.array([[4, 28, 16, 5],
    [32, 2, 15, 4],
    [1, 9, 12, 18]])
print(x.ndim)

# Three-dimensional
x = np.array([[[4, 28, 16, 5],
    [32, 2, 15, 4],
    [1, 9, 12, 18]],
    [[6, 7, 36, 25],
    [4, 20, 21, 7],
    [11, 17, 6, 2]]])
print("Rank: ", x.ndim)
print("Shape: ", x.shape)
print("Dtype: ", x.dtype)

# Mnist example
(train_img, train_labels), (test_img, test_labels) = mnist.load_data()
print("Rank: ", train_img.ndim)
print("Shape: ", train_img.shape)
print("Dtype: ", train_img.dtype)

# Example data
plt.figure()
plt.imshow(train_img[30], cmap='binary')
plt.show()

# Reshaping data
train_img = train_img.reshape(60000, 28*28)
print(train_img.shape)

train_img = train_img.reshape(60000, 28, 28)
plt.figure()
plt.imshow(train_img[30], cmap='binary')
plt.show()

# Neural pipeline
model = models.Sequential()
model.add(layers.Dense(64))
model.add(layers.Dense(128))