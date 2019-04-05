#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 01:50:57 2019

@author: ififsun
"""
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from encoders import SupervisedGraphSage
from collections import OrderedDict
from sklearn.cluster import MiniBatchKMeans
from plotModel import plotGraphStrc
from dec import DEC
from sklearn.cluster import KMeans
import torch.nn as nn
from decautoencoder import DEC_AE
from utils import target_distribution, cluster_accuracy, summary
from cluster import ClusterAssignment

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


def activeL(model, val, label, features, adj_lists, num_features, num_hidden, num_cls, xi=1e-6, eps=2.5, num_iters=10):
    #obtain the adj matrix and find the best perturbation direction then add perturbation to the attention matrix
    
    parameters = model.state_dict()
    
    '''
    predict =  F.softmax(model(val),dim = 1).data
    conf,pseudoL = torch.max(predict,  dim = 1)
    cntPreL = list(np.zeros(num_cls))
    sumPreC = list(np.zeros(num_cls))
    for i in range(len(pseudoL)):
        cntPreL[pseudoL[i]] += 1
        sumPreC[pseudoL[i]] += conf[i].double()
    
    avgpreC =  torch.FloatTensor(sumPreC)/torch.FloatTensor(cntPreL)
    #print(sumPreC,cntPreL, avgpreC )
    _, minConC = torch.min(avgpreC, dim=0)
    #print(minConC)
    sampL = list(np.array([ 1 if i == int(minConC) else 0 for i in pseudoL ]).nonzero())
    #print(sampL)
    '''
    
    
    new_state_dict = OrderedDict()
    for k, v in parameters.items():
        if k!= 'layer_1.features.weight':
            new_state_dict[k] = v
    
    actModel = SupervisedGraphSage(features,  adj_lists, num_features, num_hidden, num_cls)
    actModel.load_state_dict(new_state_dict)
    
    ul_y, para1, para2  = actModel(val, actE = True)
    
    mask1, unique_nodes_list1, unique_nodes1 = para1
    mask2, unique_nodes_list2, unique_nodes2 = para2
    
    
    

    #d = torch.Tensor(mask2.size()).normal_()
    zero_vec = torch.zeros_like(mask2)
    for i in range(num_iters):
        d = xi *_l2_normalize(mask2)
        d = Variable(d, requires_grad=True)
        
        mask2Wet = torch.where(mask2 > 0, mask2+d, zero_vec)
        y_hat = actModel(val, att = True, samL1 = (mask1, unique_nodes_list1, unique_nodes1), samL2 = (mask2Wet, unique_nodes_list2, unique_nodes2))
        delta_kl = kl_div_with_logit(ul_y, y_hat).mean(dim = 0)
        delta_kl.backward(retain_graph=True)

        d = d.grad.data.clone()
        actModel.zero_grad()

    d = _l2_normalize(d)
    d = Variable(d)
    r_adv = eps *d
    #print(r_adv)
    # compute lds
    mask2Wet = torch.where(mask2 > 0, mask2+r_adv, zero_vec)
    y_hat = actModel(val, att = True,samL1 =(mask1, unique_nodes_list1, unique_nodes1), samL2 = (mask2Wet, unique_nodes_list2, unique_nodes2))
    
    '''
    loss = kl_div_with_logit(ul_y, y_hat).mean(dim = 0)
    '''
    klDiv = kl_div_with_logit(ul_y, y_hat)
    predict =  F.softmax(y_hat,dim = 1).data
    conf,pseudoL = torch.max(predict,  dim = 1)
    
    cntPreL = list(np.zeros(num_cls))
    sumPreC = list(np.zeros(num_cls))
    sumKlD = list(np.zeros(num_cls))
    for i in range(len(pseudoL)):
        cntPreL[pseudoL[i]] += 1
        sumPreC[pseudoL[i]] += conf[i].double()
        sumKlD[pseudoL[i]] += klDiv[i].detach().double()
    
    avgpreC =  torch.FloatTensor(sumPreC)/torch.FloatTensor(cntPreL)
    avgKlD =  torch.FloatTensor(sumKlD)/torch.FloatTensor(cntPreL)
    #print(sumPreC,cntPreL, avgpreC )
    _, minConC = torch.min(avgpreC, dim=0)
    _, maxavgKLD = torch.max(avgKlD, dim=0)
    print(minConC, maxavgKLD)
    
    sampL = list(np.array([ 1 if i == int(minConC) else 0 for i in pseudoL ]).nonzero())
    #print(sampL)
    
    
    _, conf_test_nod = klDiv[sampL].topk(3)
    conf_test_nod = torch.tensor(sampL).squeeze()[conf_test_nod]
    #print(ul_y[conf_test_nod], y_hat[conf_test_nod], label[conf_test_nod],predict[conf_test_nod], pseudoL[conf_test_nod] )
    print(label[conf_test_nod], pseudoL[conf_test_nod] )
    #print(aa)
    unconf_test_nod = [i for j, i in enumerate(range(len(val))) if j not in conf_test_nod]
    
    
    return val[conf_test_nod], conf_test_nod, unconf_test_nod
               
            
def activeUnsL(model, node,  label, features, adj_lists, num_features, num_hidden, num_cls, filetime,labels, xi=1e-6, eps=2.5, num_iters=10):
    #obtain the adj matrix and find the best perturbation direction then add perturbation to the attention matrix
    
    encSpc = model(node, actE = True)
    
    dec_ae = DEC_AE(50, 100, num_hidden)
    dec = DEC(num_cls, 50, dec_ae)
    kmeans = KMeans(n_clusters=dec.cluster_number, n_init=20)
    features = []
    # form initial cluster centres
    dec.pretrain(encSpc.data)
    features = dec.ae.encoder(encSpc).detach()
    predicted = kmeans.fit_predict(features)
    predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
    _, accuracy, _, _ = cluster_accuracy( label, predicted)
    print("ACCU", accuracy)
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
    #print(features)
    dec.assignment.cluster_centers = torch.nn.Parameter(cluster_centers)
    loss_function = nn.KLDivLoss(size_average=False)
    delta_label = None
    optimizer = torch.optim.SGD(dec.parameters(), lr = 0.01, momentum=0.9)
    for epoch in range(250):
        dec.train()
        output = dec(encSpc)
        target = target_distribution(output).detach()
        loss = loss_function(output.log(), target) / output.shape[0]
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step(closure=None)
        features = dec.ae.encoder(encSpc).detach()
        #predicted =  dec(test)
        predicted = output.argmax(dim = 1)
        delta_label = float((predicted != predicted_previous ).float().sum().item()) / predicted_previous.shape[0]

        predicted_previous = predicted
        _, accuracy, _, _ = cluster_accuracy(np.array(predicted), np.array(label))
        
        if epoch % 50 == 49: 
            count_matrix = np.zeros((num_cls, num_cls), dtype=np.int64)
            for i in range(len(predicted)):
                count_matrix[np.array(predicted)[i], np.array(label)[i]] += 1
            for i in range(num_cls):
                print(count_matrix[i])
            summary(node, labels, np.array(predicted), num_cls, filetime, outlog = False, output=None)
        
            print(loss)
            print("ACCU", accuracy)
    
    
    
    #plotGraphStrc(list(node),y_pred_s.data,F.softmax(y_pred_s,dim = 1).data.numpy().argmax(axis=1) , adj_lists, time = filetime, name = 'slfT-unS')
           
    #print(y_pred_s)

def activeUnL(model, node,  label, features, adj_lists, num_features, num_hidden, num_cls, filetime,labels, xi=1e-6, eps=2.5, num_iters=10):
    #obtain the adj matrix and find the best perturbation direction then add perturbation to the attention matrix
    
    encSpc = model(node, actE = True).data
    kmeans = KMeans(n_clusters=num_cls, n_init=20)
    predicted = kmeans.fit_predict(encSpc)
    count_matrix = np.zeros((num_cls, num_cls), dtype=np.int64)
    for i in range(len(predicted)):
        count_matrix[np.array(predicted)[i], np.array(label)[i]] += 1
    for i in range(num_cls):
        print(count_matrix[i])
    #summary(node, labels, np.array(predicted), num_cls, filetime, outlog = False, output=None)
    _, accuracy, res_1, res_2 = cluster_accuracy(np.array(predicted), np.array(label))
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
    cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.float)
    DisMax = np.zeros(num_cls)
    DisNod = [0 for i in range(num_cls)]
    '''
    for j in range(len(encSpc)):
        dis = np.sqrt(((encSpc[j] - cluster_centers[int(cluster_labels[j])])**2).sum(dim = 0))
        if dis > DisMax[res_2[int(cluster_labels[j])]]:
            DisNod[res_2[int(cluster_labels[j])]] = int(node[j])
    '''


    assign_squared = torch.sqrt(torch.sum((encSpc.unsqueeze(1) - cluster_centers)**2, 2).float() )  
    
    assignTop2,_ = assign_squared.topk(2,dim  = 1)
    print(assignTop2)
    _, DisNod = abs(assignTop2[:,0]-assignTop2[:,1]).topk(7,dim = 0, largest = False )
    unconf_test_nod = [i for j, i in enumerate(node) if i not in DisNod]
    print("ACCU", accuracy)
    return DisNod, unconf_test_nod