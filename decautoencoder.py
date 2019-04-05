#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:13:29 2019

@author: ififsun
"""
import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
import torch.nn.functional as F
from scipy import stats
import random
from torch.autograd import Variable
from aggregators import MeanAggregator


class DEC_AE(nn.Module):
    def __init__(self,  num_finalfeat, num_hidden, embed_dim, pretrainMode = True, alpha = 0.2):
        super(DEC_AE,self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.num_finalfeat = num_finalfeat
        self.pretrainMode = pretrainMode
        self.fc1 = nn.Linear(embed_dim,num_hidden)
        self.fc2 = nn.Linear(num_hidden,num_finalfeat)
        self.relu = nn.LeakyReLU(0.1)
        self.fc_d1 = nn.Linear(num_hidden,embed_dim)
        self.fc_d2 = nn.Linear(num_finalfeat,num_hidden)
        self.alpha = 1.0
        #self.weight = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(num_hidden, self.num_finalfeat)))
        
    def setPretrain(self,mode):
        """To set training mode to pretrain or not, 
        so that it can control to run only the Encoder or Encoder+Decoder"""
        self.pretrainMode = mode
                
    def encoder(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        self.x_ae = x
        return self.x_ae
    def decoder(self, x_ae): 

        x = self.fc_d2(x_ae)
        x = self.relu(x)
        x = self.fc_d1(x)
        x_de = x
        return x_de