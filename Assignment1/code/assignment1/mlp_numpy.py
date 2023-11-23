################################################################################
# MIT License
#
# Copyright (c) 2023 University of Amsterdam

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal

# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *

from functools import reduce


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs: int, n_hidden: list[int], n_classes: int):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.n_hidden = n_hidden

        self.init_layers()
        #######################
        # END OF YOUR CODE    #
        #######################

    def init_layers(self):
        self.layers = reduce(self.add_hidden_layer, self.n_hidden, [])
        self.add_output_layer()

    def add_hidden_layer(self, layers: list[object], hidden_dim: int) -> list[object]:
        is_input_layer = len(layers) == 0
        in_features = self.n_inputs if is_input_layer else layers[-1].out_features

        layers.append(LinearModule(in_features, hidden_dim, input_layer=is_input_layer))
        layers.append(ELUModule())

        return layers

    def add_output_layer(self):
        last_linear_layer_idx = -2
        self.layers.append(
            LinearModule(
                self.layers[last_linear_layer_idx].out_features, 
                self.n_classes
            )
        )
        self.layers.append(SoftMaxModule())

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        batch_size, channels, width, height = x.shape
        x = x.reshape(batch_size, width * height * channels)

        out = reduce(self.feed_layer, self.layers, x)
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def feed_layer(self, inp, layer):
        return layer.forward(inp)

    def backward(self, dout, lr):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        for layer in self.layers[::-1]:
            dout = layer.backward(dout)
        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        for layer in self.layers():
            layer.clear_cache()
        #######################
        # END OF YOUR CODE    #
        #######################
