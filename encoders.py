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

    def __init__(self, features, adj_lists, feature_dim, embed_dim, num_classes, device, num_nodes = 2708, alpha = 0.2):
        super(SupervisedGraphSage, self).__init__()
        
        self.embed_dim = embed_dim
        self.device = device
        self.layer_1 = MeanAggregator(feature_dim, self.device, alpha = alpha, embed_dim = embed_dim) 
        self.layer_2 = MeanAggregator(embed_dim , self.device, alpha = alpha, embed_dim = embed_dim )
        self.adj_lists = adj_lists
        self.features = features
        
        self.xent = nn.Softmax(dim=1)
        
        

        self.weight = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(num_classes, self.embed_dim).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)))
        #init.xavier_uniform(self.weight)
    def forward(self,  nodes, num_sample = 10, gcn = True, multi = True):
        x_2 = self.layer_2(lambda nodes: self.layer_1(self.features, nodes, self.adj_lists, num_sample=5, gcn = True), nodes, self.adj_lists, num_sample=10, gcn = True) 
        '''
        else:
            mask1, unique_nodes_list1, unique_nodes1 = samL1
            mask2, unique_nodes_list2, unique_nodes2 = samL2
            x_2 = self.layer_2(lambda nodes: self.layer_1(self.features, nodes, self.adj_lists, num_sample=15, gcn = True, sap = True, mask = mask1, unique_nodes_list= unique_nodes_list1, unique_nodes= unique_nodes1), nodes, self.adj_lists, num_sample=25, gcn = True,sap = True, mask = mask2, unique_nodes_list= unique_nodes_list2, unique_nodes= unique_nodes2) 
        '''
            
       
        #print(self.weight.mm(x_2.t()).t())
        scores = self.weight.mm(x_2.t()).t()
        
        #print(scores.shape, 'score')
        self.scores = scores
        
        return scores
        
        
class SupervisedGraphSageMulti(nn.Module):

    def __init__(self, features, adj_lists, feature_dim, embed_dim, num_classes, device, num_nodes = 2708, alpha = 0.2):
        super(SupervisedGraphSageMulti, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.layer_1 = MeanAggregator(feature_dim, self.device, alpha = alpha, embed_dim = embed_dim) 
        self.layer_2 = MeanAggregator(embed_dim , self.device, alpha = alpha, embed_dim = embed_dim )
        self.adj_lists = adj_lists
        self.features = features
        
        self.xent = nn.Sigmoid()
        
        

        self.weight = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(num_classes, self.embed_dim).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)))
        #init.xavier_uniform(self.weight)
    def forward(self,  nodes, num_sample = 10, gcn = True, multi = True):
        x_2 = self.layer_2(lambda nodes: self.layer_1(self.features, nodes, self.adj_lists, num_sample=5, gcn = True), nodes, self.adj_lists, num_sample=10, gcn = True) 
        '''
        else:
            mask1, unique_nodes_list1, unique_nodes1 = samL1
            mask2, unique_nodes_list2, unique_nodes2 = samL2
            x_2 = self.layer_2(lambda nodes: self.layer_1(self.features, nodes, self.adj_lists, num_sample=15, gcn = True, sap = True, mask = mask1, unique_nodes_list= unique_nodes_list1, unique_nodes= unique_nodes1), nodes, self.adj_lists, num_sample=25, gcn = True,sap = True, mask = mask2, unique_nodes_list= unique_nodes_list2, unique_nodes= unique_nodes2) 
        '''
            
       
        #print(self.weight.mm(x_2.t()).t())
        scores = self.xent(self.weight.mm(x_2.t()).t())
        
        #print(scores.shape, 'score')
        self.scores = scores
        
        return scores
    
class GraphMultiPred(nn.Module):

    def __init__(self, features, adj_lists, feature_dim, embed_dim, num_classes, device, num_nodes = 2708, alpha = 0.2):
        super(GraphMultiPred, self).__init__()
        
        self.embed_dim = embed_dim
        self.device = device
        self.weight = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(num_classes, self.embed_dim).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)))
        #init.xavier_uniform(self.weight)
    def forward(self,  x_2, num_sample = 10, gcn = True):

        scores = self.weight.mm(x_2.t()).t()
        
        return scores
