#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:32:45 2019

@author: ififsun
"""
import torch
import torch.nn as nn
from torch.nn import init
import networkx as nx
import numpy as np
import random
from collections import defaultdict
import json
import pickle as pkl
from networkx.readwrite import json_graph
import scipy.sparse as sp
import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


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
    num_class = 0
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]
            if labels[i][0] > num_class:
                num_class = labels[i][0]
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
    val = rand_indices[:1000]
    train = list(rand_indices[1500:1640])
    other = list(rand_indices[1640:2708])
    return feat_data, labels, adj_lists, num_nodes, num_feats, train, test, val, num_class+1

def load_pubmed():
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    num_class = 0
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            if labels[i][0] > num_class:
                num_class = labels[i][0]
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
    test = rand_indices[1000:1500]
    val = rand_indices[:1000]
    train = list(rand_indices[1500:1560])
    return feat_data, labels, adj_lists, num_nodes, num_feats, train, test, val, num_class+1

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
    
    num_class = 0
    for key in class_map:
        num_class = len(np.array(class_map[key]))
    node_map = id_map

    adj_lists = defaultdict(set)
    for edge in G.edges():
        
        paper1 = node_map[str(edge[0])]
        paper2 = node_map[str(edge[1])]
        adj_lists[paper1].add(paper2)
        adj_lists[paper2].add(paper1)
        
    train = []
    test = []
    val = []
    
    for edge in G.edges():
        nod = edge[0]
        if G.node[edge[0]]['val'] == True:
            val.append(node_map[str(nod)])
        elif G.node[edge[0]]['test'] ==True:
            test.append(node_map[str(nod)])
        else:
            train.append(node_map[str(nod)])
    train, test, val = list(set(train)), list(set(test)), list(set(val))
    return feats, class_map, adj_lists, num_nodes, num_feats, train, test, val, num_class

def load_reddit():
    prefix = "example_data/reddit"

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
    
    num_class = 0
    nodes = []
    for key in class_map:
        labels[id_map[key]] = class_map[key]
        nodes.append(id_map[key])
        if class_map[key] > num_class:
            num_class = class_map[key]

    adj_lists = defaultdict(set)
    for edge in G.edges():
        
        paper1 = edge[0]
        paper2 = edge[1]
        adj_lists[paper1].add(paper2)
        adj_lists[paper2].add(paper1)
        
    train = []
    test = []
    val = []
    
    rand_indices = np.random.permutation(num_nodes)
    test = nodes[list(rand_indices[10000:15000])]
    val = nodes[list(rand_indices[:500])]
    train = nodes[list(rand_indices[15000:30000])]
    other = nodes[list(rand_indices[16400:27080])]
    train, test, val = list(set(train)), list(set(test)), list(set(val))
    return feats, labels, adj_lists, num_nodes, num_feats, train, test, val, num_class+1

def load_data(dataset, filetime):
    if dataset == 'cora':
        print('loading cora dataset')
        writetofile('loading cora dataset', "result/"+dataset+"/result_para", filetime)
        return load_cora()
    elif dataset == 'pubmed':
        print('loading pubmed dataset')
        writetofile('loading pubmed dataset', "result/"+dataset+"/result_para", filetime)
        return load_pubmed()
    elif dataset == 'ppi':
        print('loading ppi dataset')
        writetofile('loading ppi dataset', "result/"+dataset+"/result_para", filetime)
        return load_ppi()
    elif dataset == 'reddit':
        print('loading reddit dataset')
        writetofile('loading reddit dataset', "result/"+dataset+"/result_para", filetime)
        return load_reddit()
    else:
        print('Have no' + dataset +'data')
        
def sampling(train, confList , device, k = 80):
    # sampling the least confident sample to train
    confList = confList[train]
    train = np.array(train)
    _, les_conf = confList.topk(k, largest = False)
    les_conf = les_conf.data.cpu().numpy()
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

def summary(dataset, val, labels, pair, num_cls, time, outlog = False, output=None):
    true = list(np.zeros(num_cls))
    false = list(np.zeros(num_cls))
    false3 = list(np.zeros(num_cls))
    for i in range(len(val)):
        if labels[val][i] == pair[i]:
            
            true[labels[val][i][0]] += 1
        else:
            false[pair[i]]+=1
            if labels[val][i] ==3:
                false3[pair[i]]+=1
                #if outlog == True:
                    #print(output.data.numpy()[i])
                    #print(F.softmax(output, dim = 1).data.numpy()[i])
    unique, counts = np.unique(labels[val], return_counts=True)
    print(counts)
    print(true)
    print(true/counts)
    print(false)
    print(false3)
    writetofile(counts, "result/"+dataset+"/result_para", time)
    writetofile(true, "result/"+dataset+"/result_para", time)
    writetofile(true/counts, "result/"+dataset+"/result_para", time)
    

def writetofile(options, path, time):
    ISOTIMEFORMAT = '%Y-%m-%d %H:%M'
    filename = path + time.strftime(ISOTIMEFORMAT) + '.txt'
    with open(filename, 'a') as f:
        f.write(str(options) + '\n')


def sepRgtWrg(y_pred, labels):
    right = []
    wrong = []
    for i in range(len(y_pred)):
        if y_pred[i]==labels[i][0]:
            right.append(i)
        else:
            wrong.append(i)
    return right, wrong
    


def vat_loss(model, ul_x, ul_y,   xi=1e-6, eps=2.5, num_iters=1):

    # find r_adv
    ul_x = model.encoder(ul_x)
    '''
    b = np.zeros((len(ul_x), max(ul_y)+1))
    b[range(len(ul_x)), ul_y] = 1
    ul_y = torch.FloatTensor(b)
    '''
    d = torch.Tensor(ul_x.size()).normal_()
    for i in range(num_iters):
        d = xi *_l2_normalize(d)
        d = Variable(d, requires_grad=True)
        y_hat = model.predict(ul_x + d)
        delta_kl = kl_div_with_logit(ul_y, y_hat).mean(dim = 0)
        delta_kl.backward(retain_graph=True)

        d = d.grad.data.clone()
        model.zero_grad()

    d = _l2_normalize(d)
    d = Variable(d)
    r_adv = eps *d
    # compute lds
    y_hat = model.predict(ul_x + r_adv.detach())
    delta_kl = kl_div_with_logit(ul_y, y_hat).mean(dim = 0)
    return delta_kl

def kl_div_with_logit(q_logit, p_logit):
    
    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = ( q *logq).sum(dim=1)
    qlogp = ( q *logp).sum(dim=1)

    return qlogq - qlogp


def _l2_normalize(d):

    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=1)).reshape((-1, 1)) + 1e-16)
    return torch.from_numpy(d)

def vat_sel(model, ul_x, ul_y,   xi=1e-6, eps=2.5, num_iters=1):

    # find r_adv
    ul_x = model.encoder(ul_x)
    '''
    b = np.zeros((len(ul_x), max(ul_y)+1))
    b[range(len(ul_x)), ul_y] = 1
    ul_y = torch.FloatTensor(b)
    '''
    d = torch.Tensor(ul_x.size()).normal_()
    for i in range(num_iters):
        d = xi *_l2_normalize(d)
        d = Variable(d, requires_grad=True)
        y_hat = model.predict(ul_x + d)
        delta_kl = kl_div_with_logit(ul_y, y_hat).mean(dim = 0)
        delta_kl.backward(retain_graph=True)

        d = d.grad.data.clone()
        model.zero_grad()

    d = _l2_normalize(d)
    d = Variable(d)
    r_adv = eps *d
    # compute lds
    y_hat = model.predict(ul_x + r_adv.detach())
    
    _, conf_test_nod = kl_div_with_logit(ul_y, y_hat).topk(5)
    #conf_test_nod = list((kl_div_with_logit(ul_y, y_hat) <= 0.9).nonzero().numpy().squeeze())
    unconf_test_nod = [i for j, i in enumerate(range(len(ul_x))) if j not in conf_test_nod]
    return list(conf_test_nod.numpy()),unconf_test_nod 






def cluster_accuracy(y_true, y_predicted, cluster_number: Optional[int] = None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.
    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = max(y_predicted.max(), y_true.max()) + 1  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1
    reassignment = np.dstack(linear_sum_assignment(count_matrix.max() - count_matrix))[0]
    accuracy = count_matrix[reassignment[:, 0], reassignment[:, 1]].sum() / y_predicted.size
    return {item[1]: item[0] for item in reassignment}, accuracy, reassignment[:, 0], reassignment[:, 1]


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()




def plotDiagram(dataset, data, model, labels, nBins, time, multiL = 0):
    ISOTIMEFORMAT = '%Y-%m-%d %H:%M'
    logits = model(data)
    ece_criterion = _ECELoss()
    before_temperature_ece = ece_criterion(logits, labels).item()
    outputs  = F.softmax(logits, dim = 1).data
    confidence,pred = outputs.max(dim = 1)
    pred = pred.cpu().numpy()
    confidence = confidence.cpu().numpy()
    Slabels = [l[0] for _,l in sorted(zip(confidence,labels))]
    Spred = [l for _,l in sorted(zip(confidence,pred))]
    
    Sconf = sorted(confidence)
    bins = 0
    cnt = [0]*nBins
    conf = [0]*nBins
    accu = [0]*nBins
    for i in range(len(Sconf)):
        if (Sconf[i] >=  bins/nBins) and (Sconf[i]< (bins+1)/nBins):
            cnt[bins] += 1
            conf[bins] += Sconf[i]
            accu[bins] += 1 if Spred[i] == Slabels[i] else 0
        elif Sconf[i] >= (bins+1)/nBins:
            bins = int(np.floor(Sconf[i]*10))
            cnt[bins] += 1
            conf[bins] += Sconf[i]
            accu[bins] += 1 if Spred[i] == Slabels[i] else 0
        else:
            print('bins larger than confidence: outs = %.2f bins = %.2f' % (Sconf[i], bins))
    conf = [ np.array(conf)[i] / np.array(cnt)[i] if np.array(cnt)[i]>0 else 0 for i in range(nBins)]
    accu = [ np.array(accu)[i] / np.array(cnt)[i] if np.array(cnt)[i]>0 else 0 for i in range(nBins)]
    print(bins)
    f1 = plt.figure(1)
    plt.subplot(211)
    plt.hist(Sconf, bins=nBins, color = 'blue', alpha = 0.5)
    plt.xlabel('Confidence')
    plt.ylabel('# samples')
    plt.title('Conf Histogram')
    #plt.ylim(0,1)
    plt.subplot(212)
    print(cnt, accu,conf)
    plt.bar([i/nBins for i in range(nBins)],accu, width = 1/nBins, color = 'blue', alpha = 0.25)
    plt.bar([i/nBins for i in range(nBins)],conf, width = 1/nBins, color = 'red', alpha = 0.25)
    plt.xlim(0,1)
    plt.ylabel('Confidence')
    plt.xlabel('Accuracy')
    plt.title('Reliability Diagram')
    f1.savefig("result/"+dataset+"/Diagram of confidence" + time.strftime(ISOTIMEFORMAT)+ str(multiL)+".png", format="PNG")
    return before_temperature_ece
    








class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        labels = Variable(torch.LongTensor(labels).type( torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor )).squeeze().data
        
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece




























