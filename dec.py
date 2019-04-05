#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:51:56 2019

@author: ififsun
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cluster import ClusterAssignment
import random
from torch.autograd import Variable

class DEC(nn.Module):
    def __init__(
            self,
            cluster_number: int,
            hidden_dimension: int,
            ae: torch.nn.Module,
            alpha: float = 1.0):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.
        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension, input to the encoder
        :param hidden_dimension: hidden dimension, output of the encoder
        :param ae: autoencoder to use, must have .encoder attribute
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(DEC, self).__init__()
        self.ae = ae  # AutoEncoder stage
        '''
        if not hasattr(ae, 'encoder'):
            raise ValueError('Autoencoder must have a .encoder attribute.')
        '''
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(cluster_number, self.hidden_dimension, alpha)

    def forward(self, batch: torch.Tensor):
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.
        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        return self.assignment(self.ae.encoder(batch))
    
    def pretrain(self, encSpc):
        
        mseloss = nn.MSELoss()
        optimizer = torch.optim.SGD(self.ae.parameters(),lr = 0.1, momentum=0.9)
        running_loss=0.0
        for epoch in range(250):
            optimizer.zero_grad()
            x_ae = self.ae.encoder(encSpc)
            x_de = self.ae.decoder(x_ae)
            loss = mseloss(x_de,encSpc) 
            loss.backward()
            optimizer.step()
            running_loss += loss.data.numpy()
            if epoch % 50 == 49:    # print every 100 mini-batches
                print('[%d] loss: %.7f' %
                      (epoch + 1, running_loss / 100))
                running_loss = 0.0
            #now we evaluate the accuracy with AE
            self.ae.eval()
