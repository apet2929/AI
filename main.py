import math
import random

import mnist
from PIL import Image
import numpy as np


def sigmoid(arr):
    a = np.squeeze(np.asarray(arr))
    out = np.zeros(len(a))
    for i, elem in enumerate(a):
        out[i] = 1/(1 + math.pow(math.e, -elem))
    return out


def ReLU(arr):  # TODO : make fast
    a = np.squeeze(np.asarray(arr))
    # out = np.zeros(len(a))
    # for i, elem in enumerate(a):
    #     out[i] = max(0, elem)
    # return out
    return np.maximum(0, a)


def feed_forward(a, weights, biases):
    matmul = np.dot(weights, a)
    result = matmul + biases
    return sigmoid(result)


images = mnist.test_images()
input_arr = images[0].flatten() / 255

inp_nodes = len(input_arr)
layer_size = 16
output_nodes = 10
# a = matrix
# b = (
a = [
    input_arr,
    np.random.uniform(-1, 1, layer_size),
    np.random.uniform(-1, 1, layer_size),
    np.random.uniform(-1, 1, output_nodes),
]

b = [
    np.random.uniform(-1, 1, layer_size),
    np.random.uniform(-1, 1, layer_size),
    np.random.uniform(-1, 1, output_nodes),
]

weights = [np.random.uniform(-1, 1, (layer_size, inp_nodes)),
           np.random.uniform(-1, 1, (layer_size, layer_size)),
           np.random.uniform(-1, 1, (output_nodes, layer_size))]

biases = [np.random.uniform(-1, 1, layer_size), np.random.uniform(-1, 1, layer_size), np.random.uniform(-1, 1, output_nodes)]

a[1] = feed_forward(a[0], weights[0], biases[0])
a[2] = feed_forward(a[1], weights[1], biases[1])
a[3] = feed_forward(a[2], weights[2], biases[2])
print(a[1])
print(a[2])
print(a[3])

# img = Image.fromarray(a[1])
# img.save('my.png')
# img.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
