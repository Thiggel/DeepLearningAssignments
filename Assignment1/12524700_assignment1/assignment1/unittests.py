################################################################################
# MIT License
#
# Copyright (c) 2023 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
import unittest
import numpy as np

from cifar10_utils import get_cifar10
from modules import LinearModule, SoftMaxModule, CrossEntropyModule
from modules import ELUModule


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    fx = f(x)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x)
        x[ix] = oldval - h
        fxmh = f(x)
        x[ix] = oldval

        grad[ix] = (fxph - fxmh) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


class TestLosses(unittest.TestCase):

    def test_crossentropy_loss(self):
        np.random.seed(42)
        rel_error_max = 1e-5

        for test_num in range(10):
            N = np.random.choice(range(1, 100))
            C = np.random.choice(range(1, 10))
            y = np.random.randint(C, size=(N,))
            X = np.random.uniform(low=1e-2, high=1.0, size=(N, C))
            X /= X.sum(axis=1, keepdims=True)

            loss = CrossEntropyModule().forward(X, y)
            grads = CrossEntropyModule().backward(X, y)

            f = lambda _: CrossEntropyModule().forward(X, y)
            grads_num = eval_numerical_gradient(f, X, verbose=False, h=1e-5)
            self.assertLess(rel_error(grads_num, grads), rel_error_max)


class TestLayers(unittest.TestCase):

    def test_linear_backward(self):
        np.random.seed(42)
        rel_error_max = 1e-5

        for test_num in range(10):
            N = np.random.choice(range(1, 20))
            D = np.random.choice(range(1, 100))
            C = np.random.choice(range(1, 10))
            x = np.random.randn(N, D)
            dout = np.random.randn(N, C)

            layer = LinearModule(D, C)

            out = layer.forward(x)
            dx = layer.backward(dout)
            dw = layer.grads['weight']
            dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)
            dw_num = eval_numerical_gradient_array(lambda w: layer.forward(x), layer.params['weight'], dout)

            self.assertLess(rel_error(dx, dx_num), rel_error_max)
            self.assertLess(rel_error(dw, dw_num), rel_error_max)

    def test_elu_backward(self):
        np.random.seed(42)
        rel_error_max = 1e-6

        for test_num in range(10):
            N = np.random.choice(range(1, 20))
            D = np.random.choice(range(1, 100))
            x = np.random.randn(N, D)
            dout = np.random.randn(*x.shape)

            layer = ELUModule()

            _ = layer.forward(x)
            dx = layer.backward(dout)
            dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)

            self.assertLess(rel_error(dx, dx_num), rel_error_max)

    def test_softmax_backward(self):
        np.random.seed(42)
        rel_error_max = 1e-5

        for test_num in range(10):
            N = np.random.choice(range(1, 20))
            D = np.random.choice(range(1, 100))
            x = np.random.randn(N, D)
            dout = np.random.randn(*x.shape)

            layer = SoftMaxModule()

            _ = layer.forward(x)
            dx = layer.backward(dout)
            dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)

            self.assertLess(rel_error(dx, dx_num), rel_error_max)


import unittest
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import torch.nn.functional as F
import torch
from helpers import *

# Import your implementation
from train_mlp_numpy import confusion_matrix

class TestConfusionMatrix(unittest.TestCase):
    def setUp(self):
        # Set up common test data
        self.true_labels = np.random.randint(0, 10,  size=32)
        self.predictions = F.softmax(torch.randn(32, 10), dim=1).detach().numpy()
        print('setUp', self.true_labels.shape, self.predictions.shape)
        print('max', self.predictions.argmax(axis=1))

    def test_confusion_matrix_equality(self):
        # Test if your implementation matches scikit-learn
        your_conf_matrix = confusion_matrix(self.predictions, self.true_labels)
        sklearn_conf_matrix = sklearn_confusion_matrix(self.true_labels, self.predictions.argmax(axis=1))

        print('yours' + str(your_conf_matrix))
        print('sklearn' + str(sklearn_conf_matrix))

        np.testing.assert_array_equal(your_conf_matrix, sklearn_conf_matrix)

    def test_confusion_matrix_properties(self):
        # Add more specific tests or properties you want to check
        pass

if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLosses)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestLayers)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestConfusionMatrix)
    unittest.TextTestRunner(verbosity=2).run(suite)
