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

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""




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
            
    return feat_data, labels, adj_lists

def summary(val, labels, pair):
    true = [0, 0, 0, 0, 0, 0, 0]
    false = [0, 0, 0, 0, 0, 0, 0]
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

def sampling(train, confList , k = 80):
    # sampling the least confident sample to train
    confList = confList[train]
    train = np.array(train)
    _, les_conf = confList.topk(k, largest = False)
    les_conf = les_conf.data.numpy()
    batch_nodes = train [les_conf]  
    return batch_nodes, les_conf 
        
def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()
    graphsage = SupervisedGraphSage(features,  adj_lists, 1433, 180, 7)
    
    #agg1 = MeanAggregator(features,1433, 0.2, 128, cuda=True)
    #hidd1 = Hidd_Encoder( 1433, 128,agg1, gcn=True, cuda=False)
    #agg2 = MeanAggregator(lambda nodes : agg1(nodes, [adj_lists[int(node)] for node in nodes]).t(),128, 0.2,128, cuda=False)    
    #hidd2 = Hidd_Encoder(agg1.embed_dim,  128, agg2, gcn=True, cuda=False)
#    graphsage.cuda()
    xent = nn.CrossEntropyLoss()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[1000:1500]
    val = rand_indices[:500]
    train = list(rand_indices[1500:1640])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.05, momentum = 0.9)
    #encoder_scheduler = StepLR(optimizer,step_size=100,gamma=0.8)
    times = []
    loss_Data = []
    confList = Variable(torch.zeros(2708))
    #optimizer_1 = torch.optim.SGD(graphsage.w, lr=0.5)

    for batch in range(100):
        #batch_nodes = train[:120]
        #random.shuffle(train)
        batch_nodes,_ = sampling(train, confList)
        start_time = time.time()
        optimizer.zero_grad()
        scores = graphsage(batch_nodes, num_sample = 10, gcn = True)
        conf,_ = scores.max(dim = 1)
        confList[batch_nodes] = conf #update confidence
        #print(scores)
        l_los = xent(scores, Variable(torch.LongTensor(labels[np.array(batch_nodes)])).squeeze())
        
        loss = l_los 
        #graphsage.zero_grad()
        loss.backward(retain_graph=True)
        
        #print(graphsage.w, grad_norm_loss)
        optimizer.step()
        #encoder_scheduler.step(batch)
        end_time = time.time()
        times.append(end_time-start_time)
        loss_Data.append(loss.data)
        if batch%100 == 0:
            print ("Validation ACCU:", accuracy_score(labels[test], F.softmax(graphsage(test),dim = 1).data.numpy().argmax(axis=1)))
        #print (batch, loss.data[0])
    #print(sum(ada1>=0), sum(ada1<0),sum(ada2>=0), sum(ada2<0))
    #print(w[0],w[1])

    test_output =  graphsage(test)
    summary(test, labels, test_output.data.numpy().argmax(axis = 1))
    print ("Validation ACCU:", accuracy_score(labels[test], F.softmax(test_output,dim = 1).data.numpy().argmax(axis=1)))
    loss_DataSelf = []
    
    model = ModelWithTemperature(graphsage)
    temperature = model.set_temperature(val, Variable(torch.LongTensor(labels[np.array(val)])).squeeze())
    graphsage, loss_DataSelf = selfTrain(graphsage, labels,train, test, confList, temperature)
    
    
    
    
    test_output = graphsage(test)
    test_output = test_output/temperature.unsqueeze(1).expand(test_output.size(0), test_output.size(1)) 
    
    summary(test, labels, test_output.data.numpy().argmax(axis = 1))
    print ("Validation ACCU:", accuracy_score(labels[test], F.softmax(test_output,dim = 1).data.numpy().argmax(axis=1))) 
    return loss_Data+loss_DataSelf, scores, test_output, labels[np.array(batch_nodes)], labels[test], graphsage, test



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
    

def selfTrain(graphsage, labels,train, test, confList, temperature = 1, update_interval = 75, maxiter = 1000, tol = 2, batch_size = 200  ):
    
    '''
    labels_pse = np.empty((len(labels), 1), dtype=np.int64)
    test_scores = graphsage(test)
    
    y_pred = np.argmax(test_scores, axis=1)
    y_pred_last = np.copy(y_pred)
    
    
    test_conf, test_output = test_scores.max(dim = 1)
    #add_data = list((test_conf >= 0.8).nonzero().data.numpy().squeeze())
    
    labels_pse[train] = labels[train]
    labels_pse[ list(test)] = calPseLb(test_scores)
    closs = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.3)
    loss_Data = []
    for batch in range(200):

        batch_nodes, ind = sampling( train+list(test), confList, k = 600)
        
        scores = graphsage(batch_nodes)
        loss = closs(scores, Variable(torch.LongTensor(labels_pse[np.array(batch_nodes)])).squeeze())
        optimizer.zero_grad()
        loss.backward(retain_grap  h=True)
        optimizer.step()
        
        
        conf_pse,_ = scores.max(dim = 1)
        
        batch_nodes[(ind>len(train)).nonzero()[0]]
        labels_pse[batch_nodes[(ind>len(train)).nonzero()[0]]] = calPseLb(test_scores)[(ind>len(train)).nonzero()[0]]
        
        confList[batch_nodes] = conf_pse
        loss_Data.append(loss.data)
    '''   
    
    #index = 0
    #index_array = np.arange(test.shape[0])
    
    closs = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.02, weight_decay=5e-4, momentum = 0.9)
    loss_Data = []
    y_pred_last = 0
    encoder_scheduler = StepLR(optimizer,step_size=100,gamma=0.8)
    confList = Variable(torch.zeros(2708))
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:

            labels_pse = np.empty((len(labels), 1), dtype=np.int64)
            labels_pse[train] = labels[train]
            test_scores = graphsage(test)
            
            test_scores = test_scores / temperature.unsqueeze(1).expand(test_scores.size(0), test_scores.size(1))
            test_output, _ = torch.max(F.softmax(test_scores,dim = 1), dim = 1)
            y_pred = F.softmax(test_scores, dim = 1).data.numpy().argmax(axis=1)
            
            conf_test_nod = list((test_output > 0.90).nonzero().data.numpy().squeeze())
            labels_pse[ test[conf_test_nod]] = calPseLb(F.softmax(test_scores, dim = 1))[conf_test_nod]
            
            #print(test_output)
            print(len(np.array(labels[test[conf_test_nod]])))
    
            #print(test_scores, np.array(pse_label).squeeze())
            print('\nIter {}: '.format(ite), end='')
           
            print("Validation ACCU:", accuracy_score( labels[test], y_pred))
                
            # check stop criterion
            if ite >0:
                #print(y_pred)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float) / y_pred.shape[0]
                print('Fraction of documents with label changes: {} %'.format(np.round(delta_label*100, 3)))
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label <= tol/100:
                print('\nFraction: {} % < tol: {} %'.format(np.round(delta_label*100, 3), tol))
                print('Reached tolerance threshold. Stopping training.')
                break
        batch_train,_ = sampling(train, confList, k = 80)
        train_set = list(batch_train) + list(test[conf_test_nod])
        len_train = len(batch_train)
        # train on batch
        #idx = index_array[index * batch_size: min((index+1) * batch_size, test.shape[0])]
        scores = graphsage(train_set)
        scores = scores / temperature
        conf,_ = F.softmax(scores, dim = 1).max(dim = 1)
        confList[batch_train] = conf[:len_train] 
        loss = closs(scores, Variable(torch.LongTensor(labels_pse[np.array(train_set)])).squeeze())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        encoder_scheduler.step(ite)
        #index = index + 1 if (index + 1) * batch_size <= test.shape[0] else 0   
        loss_Data.append(loss.data)
    return graphsage, loss_Data

if __name__ == "__main__":
    loss_Data, scores, val_output, labels_train, labels_val, graphsage, test = run_cora() #0.862 time 0.0506  #0.888 - 0.894 avg time 0.74 # 0.846
    plt.plot(range(len(loss_Data)), loss_Data)
    #loss_Data_p = run_pubmed() #0.808  time: 0.4415 #0.832  avg time 15.62  #80.2    
    #plt.plot(range(len(loss_Data_p)), loss_Data_p) 
    
    
    
    """
    labelAdjDic= {0:[],1:[],2:[],3:[],4:[],5:[],6:[]}
for i in labelDic:
    labelAdjDic[i].append([len(adj_lists[j]) for j in labelDic[i]])
#%%
for i in labelAdjDic:
    print(np.var(labelAdjDic[i]))
    """