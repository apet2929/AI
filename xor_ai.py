import math

import numpy as np
from PIL import Image


# expand img to a vector and scale down between 0 and 1


# show img
# img = Image.fromarray(np_image)
# img.save('my.png')
# img.show()


def ReLU(arr):  # TODO
    arr = np.squeeze(np.asarray(arr))
    new_arr = np.zeros(len(arr))
    for i, n in enumerate(arr):
        new_arr[i] = max(0.0, n)
    return new_arr
    # return np.maximum()


def sigmoid(arr):
    # a = np.squeeze(np.asarray(arr))
    # out = np.zeros(len(a))
    # for i, elem in enumerate(a):
    #     out[i] = 1 / (1 + math.pow(math.e, -elem))
    # return out
    return 1 / (1 + math.pow(math.e, -arr))


def feed_forward(nodes, weights, basis):
    return sigmoid(np.dot(weights, nodes) + basis)


def cost(output, ans):
    return np.sum(np.square(output - ans))


def predict():
    a[1] = np.array([feed_forward(a[0], W[0], b[0])])
    a[2] = np.array([feed_forward(a[1], W[1], b[1])])





# img = Image.fromarray(a[1])
# img.save('my.png')
# img.show()

current_layer = 1  # stores the current layer being backprop'd

def train(inputs, answers):
    avg_grad = []
    grads = []
    for i in range(len(inputs)):
        grads.append(calc_grad(inputs[i], answers[i]))

    for i in range(len(grads)):
        for j in range(len(grads[i])):
            avg_grad[j] += grads[i][j]

    for i in range(len(avg_grad)):
        avg_grad[i] /= len(grads)

    W[1][0] -= avg_grad[0]
    b[1][0] -= avg_grad[1]
    W[0][0] -= avg_grad[2]
    W[0][1] -= avg_grad[3]
    b[0][0] -= avg_grad[4]



# The derivative of the sigmoid function with respect to z
def calc_grad(_input, ans):
    global current_layer
    current_layer = 2
    predict()
    grad_vector = []
    dc_da = 2 * (np.subtract(a[current_layer], ans))[0]
    for i in range(2, -1, -1):
        current_layer = i
        dc_dw, dc_db, dc_da = pseudo_grad(dc_da)
        grad_vector.extend(dc_dw)
        grad_vector.extend(dc_db)

    return grad_vector


def weights(layer_num):
    return W[layer_num-1]


def biases(layer_num):
    return b[layer_num-1]


def nodes(layer_num):
    return a[layer_num]


def pseudo_grad(dc_da):
    dc_dw = []
    dc_db = []
    # TODO: Figure out which weights line up with which nodes
    # nodes.size always equals weights.size + 1
    # resolve the indexing conflict (maybe make a function to translate between index and layer # for weights and biases?)
    for i in range(a[current_layer].size):
        #  for every node in the current layer, there exists:
        #  1 dc_db
        #  1 dc_dw for every node in the previous layer
        prev_nodes = []

        z = nodes(current_layer)[i]
        prev_nodes = nodes(current_layer-1)
        dc_db.append(dsigmoid(z) * dc_da)
        dc_dw.extend(prev_nodes * dc_db[0])
        dc_da = np.sum(weights(current_layer)) * dsigmoid(z) * 2 * dc_da

    return dc_dw, dc_db, dc_da


def dsigmoid(z: float) -> float:
    return sigmoid(z) * sigmoid(1 - z)

inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
expected = [0, 1, 1, 0]

a = [
    np.array([0, 0]),  # inputs
    np.array(np.array([0])),  # hiddenlayer
    np.array(np.array([0])) # output
]

W = [
    # np.array([0]),
    np.random.uniform(-1, 1, 2),
    np.random.uniform(-1, 1, 1)
]

b = [
    np.random.uniform(-1, 1, 1),  # hidden layer
    np.random.uniform(-1, 1, 1)  # output layer
]

# a[0] = np.array([0, 0])
# print("a= " + str(a))
# print("w= " + str(W))
# # predict()a[2]
#
# print("output:", a[2])
# print("cost:", cost(a[2], np.array([0])))
# print("grad: ", calc_grad(inputs[0], expected[0]))

for j in range(20):
    for i in range(len(inputs)):
        a[0] = inputs[i]
        predict()
        c = cost(a[2][0], expected[1])
        print(f"i={i}, cost i={c}")
    train(inputs, expected)
    print()

