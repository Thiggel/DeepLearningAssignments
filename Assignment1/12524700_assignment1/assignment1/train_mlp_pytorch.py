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
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helpers import save_loss_curve


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    n_classes = predictions.shape[1]
    conf_mat = torch.zeros((n_classes, n_classes), dtype=torch.int32)

    for true_label, pred_label in zip(targets, predictions.argmax(axis=1)):
        conf_mat[true_label, pred_label] += 1
    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    true_positives = torch.diag(confusion_matrix)
    false_negatives = torch.sum(confusion_matrix, axis=0) - true_positives
    false_positives = torch.sum(confusion_matrix, axis=1) - true_positives

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (false_positives + false_negatives)

    metrics = {
        'accuracy': true_positives.sum() / confusion_matrix.sum().sum(),
        'precision': precision,
        'recall': recall,
        'f1_beta':  (1 + beta**2) * precision * recall / (beta**2 * precision + recall),
    }
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10, plot_conf_mat=False):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    sum_conf_mat = None
    for (inputs, targets) in tqdm(data_loader):
        predictions = model(inputs)
        conf_mat = confusion_matrix(predictions, targets)
        if sum_conf_mat is None:
            sum_conf_mat = conf_mat
        else:
            sum_conf_mat += conf_mat

    if plot_conf_mat:
        save_conf_mat(sum_conf_mat)

    metrics = confusion_matrix_to_metrics(sum_conf_mat)
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def save_conf_mat(conf_mat):
    plt.matshow(conf_mat)
    plt.colorbar()
    plt.savefig('conf_mat.png')


def train_epoch(cifar10_loader, model, loss_module, optimizer):
    loss_sum = 0
    for (inputs, targets) in tqdm(cifar10_loader['train']):
        optimizer.zero_grad()

        predictions = model(inputs)

        loss = loss_module(predictions, targets)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    loss_mean = loss_sum / len(cifar10_loader['train'])

    return loss_mean


def f1(precision, recall, beta=1.):
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def print_metrics(metrics):
    precision = metrics["precision"]
    recall = metrics["recall"]

    print(f'accuracy: {metrics["accuracy"]}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1 for beta=0.1: {f1(precision, recall, beta=0.1)}')
    print(f'f1 for beta=1: {metrics["f1_beta"]}')
    print(f'f1 for beta=10: {f1(precision, recall, beta=10)}')


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    val_accuracies = []
    train_losses = []

    # TODO: Initialize model and loss module
    model = MLP(n_inputs=3 * 32 * 32, n_hidden=hidden_dims, n_classes=10)
    loss_module = nn.CrossEntropyLoss()
    best_model = None
    best_val_acc = 0
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # TODO: Training loop including validation
    for epoch in range(epochs):
        train_loss = train_epoch(cifar10_loader, model, loss_module, optimizer)
        train_losses.append(train_loss)
        val_acc = evaluate_model(model, cifar10_loader['validation'])['accuracy']
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = deepcopy(model)

        print(f'Epoch: {epoch + 1}, Train Loss: {train_loss}, Validation Accuracy: {val_acc}')

    # TODO: Test best model
    metrics = evaluate_model(best_model, cifar10_loader['test'], plot_conf_mat=True)
    test_accuracy = metrics['accuracy']
    print('Test Accuracy:', test_accuracy)

    print_metrics(metrics)

    # TODO: Add any information you might want to save for plotting
    logging_info = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
    }

    save_loss_curve(train_losses, val_accuracies, epochs, filename = 'loss_curve_pytorch.png')
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    
