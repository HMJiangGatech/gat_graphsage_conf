#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:32:45 2019

@author: ififsun
"""
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import networkx as nx
import numpy as np
import time
import random
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
import torch.nn.functional as F
from aggregators import MeanAggregator
from encoders import SupervisedGraphSage
import matplotlib.pyplot as plt
from math import log
from temperature_scaling import ModelWithTemperature
import numpy as np
import random
import json
import sys
import os
import pickle as pkl
import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp




class SupervisedGraphWeight(nn.Module): # train weight

    def __init__(self, num_classes, enc):
        super(SupervisedGraphWeight, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        
        embeds = self.enc(nodes)
        #print(embeds.shape, 'embeds2')
        
        scores = self.weight.mm(embeds)

        return scores.t()

    def loss(self, nodes, labels):
        
        scores = self.forward(nodes)
        #print(self.xent(scores, labels.squeeze()), 'scores')
        return self.xent(scores, labels.squeeze())
def load_cite():
    #hardcoded for simplicity...
    num_nodes = 3327
    num_feats = 3703
    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    path = "citeseer"
    for i in range(len(names)):
        with open("{}/ind.citeseer.{}".format(path, names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding="latin1"))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    #Citeseer dataset contains some isolated nodes in the graph

    features = sp.vstack((x, tx)).tolil()
    features = torch.FloatTensor(np.array(features.todense()))
    
    adj_lists = graph
    
    labels = np.empty((num_nodes,1), dtype=np.int64)
    
    for i in range(len(x)):
        labels[i] = [np.array(y)[i].argmax()]
    for i in range(len(tx)):
        labels[i] = [np.array(ty)[i].argmax()]
    train = list(range(len(x)))
    test = list(range(len(x),len(x)+len(tx)))
    #val = 
    return feat_data, labels, adj_lists, num_nodes, num_feats, train, test, val

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[1000:1500]
    val = rand_indices[:500]
    train = list(rand_indices[1500:1640])
    return feat_data, labels, adj_lists, num_nodes, num_feats, train, test, val

def load_pubmed():
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    np.random.seed(1)
    random.seed(1)        
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:1600])
    return feat_data, labels, adj_lists, num_nodes, num_feats, train, test, val

def load_ppi():
    prefix = "example_data/ppi"

    feats = np.load(prefix + "-feats.npy")
    num_nodes = len(feats)
    num_feats = len(feats[0])
    class_map = json.load(open(prefix + "-class_map.json"))
    id_map = json.load(open(prefix + "-id_map.json"))
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n
        
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    
    labels = np.empty((num_nodes,1), dtype=np.int64)
    
    for key in class_map:
        labels[int(key)] = [np.array(class_map[key]).argmax()]
    node_map = id_map

    adj_lists = defaultdict(set)
    for edge in G.edges():
        paper1 = node_map[edge[0]]
        paper2 = node_map[edge[1]]
        adj_lists[paper1].add(paper2)
        adj_lists[paper2].add(paper1)
        
    train = []
    test = []
    val = []
    
    for edge in G.edges():
        nod = edge[0]
        if G.node[edge[0]]['val'] == True:
            val.append(node_map[nod])
        elif G.node[edge[0]]['test'] ==True:
            test.append(node_map[nod])
        else:
            train.append(node_map[nod])
    train, test, val = list(set(train))[:300], list(set(test)), list(set(val))  
          
    return feats, labels, adj_lists, num_nodes, num_feats, train, test, val

def load_data(dataset):
    if dataset == 'cora':
        return load_cora()
    elif dataset == 'pubmed':
        return load_pubmed()
    elif dataset == 'ppi':
        return load_ppi()
    else:
        print('Have no' + dataset +'data')
        
def sampling(train, confList , k = 80):
    # sampling the least confident sample to train
    confList = confList[train]
    train = np.array(train)
    _, les_conf = confList.topk(k, largest = False)
    les_conf = les_conf.data.numpy()
    batch_nodes = train [les_conf]  
    return batch_nodes, les_conf 

def calPseLb(q, power=2):
    '''
    n = scores.shape[0]
    m = scores.shape[1]
    pseScores = torch.zeros(n, m)
    f = torch.sum(scores, dim = 0)
    for i in range(scores.shape[0]):
        norm = 0
        for j in range(scores.shape[1]):
            norm += scores[i][j]**2/f[j]
        for j in range(scores.shape[1]):
            pseScores[i][j] = (scores[i][j]**2/f[j])/norm
    pseLb = pseScores.data.numpy().argmax(axis = 1)
    return np.expand_dims(pseLb,axis = 1)  

    def target_distribution(self, q, power=2):
    
    weight = q**power / torch.sum(q, dim=0)
    p = (weight.t() /torch.sum( weight, dim=1)).t()
    pseLb = p.data.numpy().argmax(axis = 1)
    print(pseLb)
    return np.expand_dims(pseLb,axis = 1)  
    '''
    return np.expand_dims(q.data.numpy().argmax(axis = 1), axis = 1)

def summary(val, labels, pair, num_cls):
    true = list(np.zeros(num_cls))
    false = list(np.zeros(num_cls))
    for i in range(len(val)):
        if labels[val][i] == pair[i]:
            
            true[labels[val][i][0]] += 1
        else:
            false[pair[i]]+=1
    unique, counts = np.unique(labels[val], return_counts=True)
    print(counts)
    print(true)
    print(true/counts)
    print(false)
