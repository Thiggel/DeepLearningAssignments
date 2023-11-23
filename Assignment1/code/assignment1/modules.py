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
"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
from helpers import to_one_hot

import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.in_features = in_features
        self.out_features = out_features
        self.last_input = None

        self.kaiming_init(input_layer)
        #######################
        # END OF YOUR CODE    #
        #######################

    def kaiming_init(self, input_layer):
        factor = 1 if input_layer else 2
        self.params['weight'] = np.random.normal(
            0, 
            np.sqrt(factor / self.in_features),
            (self.out_features, self.in_features)
        )
        self.params['bias'] = np.zeros(self.out_features)

    def gradient_descent(self, lr):
        self.params['weight'] -= lr * self.grads['weight']
        self.params['bias'] -= lr * self.grads['bias']

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # (batch_size, in_features) x (in_features, out_features) + (batch_size, out_features) -> (batch_size, out_features)
        out = x @ self.params['weight'].T + np.tile(self.params['bias'], (x.shape[0], 1))

        self.last_input = x
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        assert self.last_input is not None, "There was no forward pass"

        # dout has dimensionality (batch_size, out_features)
        # last_input has dimensionality (batch_size, in_features)

        # dW has dimensionality (in_features, out_features)
        self.grads['weight'] = dout.T @ self.last_input

        # db has dimensionality (out_features)
        batch_size = self.last_input.shape[0]
        self.grads['bias'] = np.ones(batch_size) @ dout

        # gradient descent
        # self.params['weight'] -= lr * self.grads['weight']
        # self.params['bias'] -= lr * self.grads['bias']

        dx = dout @ self.params['weight']
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it
        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.last_input = None
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.last_input = x

        out = np.where(x > 0, x, np.exp(x) - 1)
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout, *args):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        assert self.last_input is not None, "There was no forward pass"

        dx = dout * np.where(self.last_input > 0, 1, np.exp(self.last_input))
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.last_input = None
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        e_x = np.exp(x - np.max(x))
        small_const_avoid_div_zero = 1e-15
        sum = e_x.sum(axis=1, keepdims=True) + small_const_avoid_div_zero
        out = self.last_output = e_x / sum
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout, *args):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        assert self.last_output is not None, "There was no forward pass"

        # last_output has dim (batch_size, num_classes)
        out_features = self.last_output.shape[1]
        dx = self.last_output * (dout - (dout * self.last_output) @ np.ones((out_features, out_features)))
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.last_output = None
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        small_const_avoid_log_zero = 1e-15
        log_x = np.log(x + small_const_avoid_log_zero)

        _, n_classes = x.shape
        y_one_hot = to_one_hot(y, n_classes)

        out = -1 * (y_one_hot * log_x).sum(axis=1).mean()
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        batch_size, n_classes = x.shape

        small_const_avoid_div_zero = 1e-15
        x += small_const_avoid_div_zero

        dx = -1 / batch_size * to_one_hot(y, n_classes) / x
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx
