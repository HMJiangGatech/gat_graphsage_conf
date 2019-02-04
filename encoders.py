import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
import torch.nn.functional as F
from scipy import stats
import random
from torch.autograd import Variable
from aggregators import MeanAggregator


class SupervisedGraphSage(nn.Module):

    def __init__(self, features, adj_lists, feature_dim, embed_dim, num_classes,num_nodes = 2708, alpha = 0.2):
        super(SupervisedGraphSage, self).__init__()
        
        self.embed_dim = embed_dim
        self.layer_1 = MeanAggregator(feature_dim, alpha = alpha, embed_dim = embed_dim, embed_attg = feature_dim) 
        self.layer_2 = MeanAggregator(embed_dim , alpha = alpha, embed_dim = embed_dim, embed_attg = feature_dim )
        self.adj_lists = adj_lists
        self.features = features
        
        self.xent = nn.Softmax(dim=1)
        
        

        self.weight = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(num_classes, self.embed_dim)))
        #init.xavier_uniform(self.weight)

        self.w = nn.Parameter(torch.FloatTensor([2, 1]), requires_grad=True)
        

    def forward(self,  nodes, num_sample = 10, gcn = True):
        
        
        x_2 = self.layer_2(lambda nodes: self.layer_1(self.features, nodes, self.adj_lists, num_sample=15, gcn = True, att_Weight= None), nodes, self.adj_lists, num_sample=25, gcn = True, att_Weight= None) 
        
       
        #print(self.weight.mm(x_2.t()).t())
        scores = self.weight.mm(x_2.t()).t()
        

        
        #print(scores.shape, 'score')

        return scores
    
    
    
