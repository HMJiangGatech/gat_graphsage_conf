import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from scipy import stats
import random
from torch.nn import init

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, feature_dim, alpha, embed_dim, embed_attg, cuda=False, gcn=True): 
        """
        Initializes the aggregator for a specific graph.
        feature_dim -- feature dimension
        alpha -- LeakyReLU
        embed_dim -- 
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.feature_dim = feature_dim
        self.cuda = cuda
        self.gcn = gcn
        self.alpha = alpha
        self.dropout = 0. 
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(
                nn.init.xavier_normal_(torch.Tensor(embed_dim, self.feature_dim if self.gcn else 2 * self.feature_dim)))
        self.weight_gAtt = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feature_dim if self.gcn else 2 * self.feature_dim))
        self.weight_all = nn.Parameter(
                torch.FloatTensor(embed_dim, embed_dim*2 if self.gcn else 2 * self.feature_dim))
        self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(feature_dim, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(feature_dim, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.linear = nn.Linear(embed_dim ,self.feature_dim*2 )
        self.fc1 = nn.LeakyReLU(self.alpha)
        self.fc2 = nn.LeakyReLU(self.alpha)
        self.fc_all = nn.LeakyReLU(self.alpha)  
        self.bn1 = nn.BatchNorm1d(num_features=embed_dim)
        #self.adaptive = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(feature_dim, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
    def sampling(self, nodes, adj_lists, num_sample = 10, gcn = True):
    
        to_neighs = [adj_lists[int(node)] for node in nodes]
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
    
        if gcn:
            samp_neighs = [samp_neigh.union((set([int(nodes[i])]))) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        #nodes index dict
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        #mask ajdancecy matrix
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        
        mask[row_indices, column_indices] = 1
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        #nodes_indexs = [i for i, e in enumerate(unique_nodes_list) if e in nodes ]
        return mask, unique_nodes_list, unique_nodes

        
    def forward(self, features, nodes, adj_lists,  num_sample=10, gcn = True, att_Weight= None):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        
        # Local pointers to functions (speed hack)
        self.features = features
        mask, unique_nodes_list, unique_nodes = self.sampling(nodes, adj_lists)
        #print(mask.shape,'mask')
        #print(len(nodes),'nodes')
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        
        f_1 = torch.matmul(self.features(torch.LongTensor(unique_nodes_list)), self.a1)
        f_2 = torch.matmul(self.features(torch.LongTensor(unique_nodes_list)), self.a2)
        
        
        e = self.leakyrelu(f_1 + f_2.transpose(0,1))
        nodes_idx = []
        if type(nodes) is torch.Tensor:
            nodesData = nodes.data.numpy()
        else:
            nodesData = nodes
        nodes_idx= [unique_nodes[n ] for n in nodesData]
        e = e[ nodes_idx ]
        zero_vec = -9e15*torch.ones_like(e)
        
        attention = torch.where(mask > 0, mask*e, zero_vec)
        attention = F.softmax(attention, dim=1)
        #attention = F.dropout(attention, self.dropout)
        
        
        
        #print(attention.shape, 'attention')
        to_feats = attention.mm(embed_matrix)
        
        
        combined = self.fc1(   self.weight.mm(to_feats.t()))
        
        '''
        
        
        to_feats_g = att_graph.mm(embed_matrix)
        combined_g = self.fc2(self.weight_gAtt.mm(to_feats_g.t()))
        #print(combined.shape, 'combined')
        combined_all = torch.cat([combined.t(), combined_g.t()],dim = 1)
        #print(combined_all.shape, 'combined_all')
        
        combined_all = self.fc_all(self.weight_all.mm(combined_all.t()))
        '''
        self.embed_matrix = embed_matrix
        
        self.mask, self.unique_nodes_list, self.unique_nodes = mask, unique_nodes_list, unique_nodes
        #print(to_feats.shape,'to')
        #print(combined.shape, 'combined')
        return combined.t()
    def ret_embedM(self,nodes ):
        return self.embed_matrix, self.mask, self.unique_nodes_list, self.unique_nodes
