import math
import pytest

import numpy as np

from multilayer_ai import NeuralNetwork


def set1():
    nn = NeuralNetwork()
    nn.a[0] = np.array([0, 0])
    nn.W = [None, np.zeros((2, 2)), np.zeros((2, 2))]
    nn.b = [None, np.zeros(2), np.zeros(2)]
    return nn


def set2():
    nn = NeuralNetwork()
    nn.a[0] = np.array([1, 1])
    nn.W = [None, np.ones((2, 2)), np.ones((2, 2))]
    nn.b = [None, np.zeros(2), np.zeros(2)]
    return nn


def set3():
    nn = NeuralNetwork()
    nn.a[0] = np.array([0.5, 0.5])
    nn.W = [
        None,
        np.asmatrix([[0.5, 0.5], [0.5, 0.5]]),
        np.asmatrix([[0.5, 0.5], [0.5, 0.5]]),
    ]
    nn.b = [
        None,
        np.asmatrix([0.5, 0.5]),
        np.asmatrix([0.5, 0.5]),
    ]
    return nn


def set4():
    nn = NeuralNetwork()
    nn.a[0] = np.array([0.69, 0.420])
    nn.W = [
        None,
        np.asmatrix([[0.1, 0.3], [0.5, 0.7]]),
        np.asmatrix([[0.2, 0.4], [0.6, 0.8]]),
    ]
    nn.b = [
        None,
        np.asmatrix([0.1, 0.2]),
        np.asmatrix([0.5, 0.6]),
    ]
    return nn


def sig(n):
    return 1 / (1 + math.exp(-n))


class TestNeuralNetwork():
    def setup(self):
        self.nn = NeuralNetwork()
        self.nn.a[0] = np.array([0, 0])
        self.nn.W = [None, np.zeros((2, 2)), np.zeros((2, 2))]
        self.nn.b = [None, np.zeros(2), np.zeros(2)]


class TestFunc(TestNeuralNetwork):
    def test_sigmoid(self):
        assert self.nn.sigmoid(np.array([0, 0])) == np.array([0.5, 0.5])
        assert self.nn.sigmoid(np.array([1, 1]) == np.array([0.731058578630074, 0.731058578630074]))
        # assert self.nn.sigmoid(0.5) == 0.6224593312018959
        # assert self.nn.sigmoid(0.11) == 0.5274723043446033

    def test_cost(self):
        assert self.nn.cost(np.array([0, 0]), np.array([0, 0])) == 0
        assert self.nn.cost(np.array([1, 1]), np.array([0, 0])) == 2
        assert self.nn.cost(np.array([0.5, 0.5]), np.array([1, 1])) == 0.5
        assert self.nn.cost(np.array([0.69, 0.420]), np.array([0.99, 0.1])) == 0.1924

    def test_dsigmoid(self):
        pass
        # assert np.array_equal(self.nn.dsigmoid(0), np.array([0.5, 0.5]))
        # assert np.array_equal(self.nn.dsigmoid(1), np.array([0.5, 0.5]))
        # assert np.array_equal(self.nn.dsigmoid(0.5), np.array([0.5, 0.5]))
        # assert np.array_equal(self.nn.dsigmoid(0.69), np.array([0.5, 0.5]))


class TestFeedForward(TestNeuralNetwork):
    def test_feed_forward(self):
        nn = set1()


class TestPredict(TestNeuralNetwork):
    def test_predict(self):
        # initializes with all 0s
        assert np.array_equal(self.nn.predict(), np.array([0.5, 0.5]))

    def test_predict_zeros(self):
        self.nn = set1()
        assert np.array_equal(self.nn.predict(), np.array([0.5, 0.5]))

    def test_predict_ones(self):
        self.nn = set2()
        expected = sig(sig(1) * 2)
        print(expected)
        actual = self.nn.predict()
        print(actual)

        assert np.array_equal(actual, np.array([expected, expected]))

    def test_predict_uni(self):
        self.nn = set3()
        expected = sig(2 * sig(1.5) + 0.5)
        assert np.array_equal(self.nn.predict(), np.array([expected, expected]))

    def test_predict_random(self):
        self.nn = set4()
        expected = [sig((0.5 * sig(0.295)) + (0.7 * sig(0.606)) + 0.2),
                    sig((0.6 * sig(0.295)) + (0.8 * sig(0.606)) + 0.6)]
        assert np.array_equal(self.nn.predict(), np.array([expected, expected]))


class TestGradDescOneLayer(TestNeuralNetwork):
    def test_grad_desc_one_layer(self):
        pass


class TestGradientDescent(TestNeuralNetwork):
    def test_grad_desc(self):
        pass

