#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:52:09 2019

@author: ififsun
"""

import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn

from utils import target_distribution, cluster_accuracy


def train(
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        node):
    """
    Train the DEC model given a dataset, a model instance and various configuration parameters.
    :param dataset: instance of Dataset to use for training
    :param model: instance of DEC model to train
    :param epochs: number of training epochs
    :param optimizer: instance of optimizer to use
    :return: None
    """
    kmeans = KMeans(n_clusters=model.cluster_number, n_init=20)
    model.train()
    features = []
    # form initial cluster centres
    features.append(model.ae.encoder(node).detach())
    actual = torch.cat(node).long()
    predicted = kmeans.fit_predict(torch.cat(features).numpy())
    predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
    _, accuracy = cluster_accuracy( actual, predicted)
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
    model.assignment.cluster_centers = torch.nn.Parameter(cluster_centers)
    loss_function = nn.KLDivLoss(size_average=False)
    delta_label = None
    optimizer = torch.optim.SGD()
    for epoch in range(150):
        features = []
        model.train()
        output = model(node)
        target = target_distribution(output).detach()
        loss = loss_function(output.log(), target) / output.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(closure=None)
        features.append(model.ae.encoder(node).detach().cpu())
        predicted, actual = predict(dataset, model)
        delta_label = float((predicted != predicted_previous).float().sum().item()) / predicted_previous.shape[0]
        predicted_previous = predicted
        _, accuracy = cluster_accuracy(predicted.numpy(), actual.numpy())


def predict(
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module):
    """
    Predict clusters for a dataset given a DEC model instance and various configuration parameters.
    :param dataset: instance of Dataset to use for training
    :param model: instance of DEC model to predict
    :param batch_size: size of the batch to predict with, default 1024
    :param cuda: whether CUDA is used, defaults to True
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param return_actual: return actual values, if present in the Dataset
    :return: tuple of prediction and actual if return_actual is True otherwise prediction
    """
    features = []
    actual = []
    model.eval()
    actual.append(value)
    features.append(model(batch).detach())  # move to the CPU to prevent out of memory on the GPU
    return torch.cat(features).max(1)[1]