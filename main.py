import math
import random

import mnist
from PIL import Image
import numpy as np


def sigmoid(arr):
    a = np.squeeze(np.asarray(arr))
    out = np.zeros(len(a))
    for i, elem in enumerate(a):
        out[i] = 1 / (1 + math.pow(math.e, -elem))
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


def cost(output, expected):
    return np.square(output - expected)

current_image_index = 0
images = mnist.test_images()
input_arr = images[0].flatten() / 255
labels_arr = mnist.test_labels()

inp_nodes = len(input_arr)
layer_size = 16
output_nodes = 10
a = [
    input_arr,
    np.random.uniform(-1, 1, layer_size),
    np.random.uniform(-1, 1, layer_size),
    np.random.uniform(-1, 1, output_nodes),
]

weights = [np.random.uniform(-1, 1, (layer_size, inp_nodes)),
           np.random.uniform(-1, 1, (layer_size, layer_size)),
           np.random.uniform(-1, 1, (output_nodes, layer_size))]

biases = [np.random.uniform(-1, 1, layer_size), np.random.uniform(-1, 1, layer_size),
          np.random.uniform(-1, 1, output_nodes)]

def predict(image):
    a[1] = feed_forward(image, weights[0], biases[0])
    a[2] = feed_forward(a[1], weights[1], biases[1])
    a[3] = feed_forward(a[2], weights[2], biases[2])
predict(a[0])
# print(a[1])
# print(a[2])
# print(a[3])
print(cost(a[3], np.asarray([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])))

# img = Image.fromarray(a[1])
# img.save('my.png')
# img.show()

current_layer = 0  # stores the current layer being backprop'd


# function to start backpropagation on the whole model
# TODO : shuffle input for training
def train(batch):
    error = 0
    for i, image in enumerate(batch):
        output = predict(image)

        # =grad_desc(dC_da_first())
        for i in range(3):
            # =grad_desc(dC_da_gen())

        # error +=
    error /= len(batch)




# man func for calculating the gradient descent at layer L
def grad_desc(dC_da):
    # dc_db = dC_db()
    # grad = [
    # dc_db
    # some_num*dc_db
    # ]
    num_of_weights = len(weights[current_layer])
    num_of_biases = len(biases[current_layer])
    out = np.zeros(len(num_of_weights + num_of_biases))
    # pre_calculated = da/dz * dC/da
    # TODO : Reference stored values of z calculated while running
    _z = z(a[current_layer], weights[current_layer], biases[current_layer])
    pre_calculated = dsigmoid(z) * dC_da
    for i, w in enumerate(weights[current_layer]):
        out[2 * i] = dC_dw(pre_calculated)
    for i, b in enumerate(biases[current_layer]):
        out[2 * i + 1] = dC_db(pre_calculated)  # might overlap

    return out


# nodes, expected = np.array()
def dC_da_gen(nodes, pre_calculated) -> float:
    s = 0
    for i in range(len(nodes)):
        pass


# nodes, expected = np.array()
def dC_da_first(nodes, expected) -> float:
    return 2 * (nodes - expected)


# calculates dC/dw
def dC_dw(pre_calculated) -> float:
    return dz_dw(a[current_layer]) * pre_calculated


# calculates dz/dw
def dz_dw(nodes) -> float:
    s = 0
    for node in nodes:
        s += node
    return s


def dC_db(pre_calculated: float) -> float:
    return pre_calculated


# iterates through nodes and weights to calc z, which is passed through sigmoid to calculate the next nodes value
# nodes, weights = np.array()
def z(nodes, weights, bias: float) -> float:
    z = 0
    for i in range(len(nodes)):
        z += nodes[i] * weights[i]
    z += bias
    return z


# The derivative of the sigmoid function with respect to z
def dsigmoid(z: float) -> float:
    return sigmoid(z) * sigmoid(1 - z)
