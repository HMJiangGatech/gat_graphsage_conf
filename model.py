import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
import random
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from encoders import SupervisedGraphSage
import matplotlib.pyplot as plt
from temperature_scaling import ModelWithTemperature
from opts import TrainOptions
from utils import *
"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""


        
def run_cora():
    opt = TrainOptions().parse()
    np.random.seed(1)
    random.seed(1)
    num_cls = opt.num_cls
    num_hidden = opt.num_hidden
    feat_data, labels, adj_lists, num_nodes,num_features, train, test, val  = load_data(opt.dataset)
    features = nn.Embedding(num_nodes, num_features)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()
    graphsage = SupervisedGraphSage(features,  adj_lists, num_features, num_hidden, num_cls)
    
    xent = nn.CrossEntropyLoss()


    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr= opt.lr_pre, momentum = opt.momentum_pre)
    #encoder_scheduler = StepLR(optimizer,step_size=100,gamma=0.8)
    times = []
    loss_Data = []
    confList = Variable(torch.zeros(num_nodes))
    #optimizer_1 = torch.optim.SGD(graphsage.w, lr=0.5
    for batch in range(opt.epoch):
        #batch_nodes = train[:55]
        #random.shuffle(train)
        batch_nodes,_ = sampling(train, confList, k = opt.k)
        start_time = time.time()
        optimizer.zero_grad()
        scores = graphsage(batch_nodes, num_sample = 10, gcn = True)
        conf,_ = scores.max(dim = 1)
        confList[batch_nodes] = conf #update confidence
        
        l_los = xent(scores, Variable(torch.LongTensor(labels[np.array(batch_nodes)])).squeeze())
        
        loss = l_los 
        
        loss.backward(retain_graph=True)
        
        
        optimizer.step()
        #encoder_scheduler.step(batch)
        end_time = time.time()
        times.append(end_time-start_time)
        loss_Data.append(loss.data)
        if batch%100 == 0:
            print ("Validation ACCU:", accuracy_score(labels[test], F.softmax(graphsage(test),dim = 1).data.numpy().argmax(axis=1)))
        #print (batch, loss.data[0])

    test_output =  graphsage(test)
    summary(test, labels, test_output.data.numpy().argmax(axis = 1), num_cls)
    print ("Validation ACCU:", accuracy_score(labels[test], F.softmax(test_output,dim = 1).data.numpy().argmax(axis=1)))
    loss_DataSelf = []
    
    model = ModelWithTemperature(graphsage)
    temperature = model.set_temperature(val, Variable(torch.LongTensor(labels[np.array(val)])).squeeze())
    graphsage, loss_DataSelf = selfTrain(graphsage, labels,train, test, confList, temperature)
    
    
    
    
    test_output = graphsage(test)
    test_output = test_output/temperature.unsqueeze(1).expand(test_output.size(0), test_output.size(1)) 
    
    summary(test, labels, test_output.data.numpy().argmax(axis = 1), num_cls)
    print ("Validation ACCU:", accuracy_score(labels[test], F.softmax(test_output,dim = 1).data.numpy().argmax(axis=1))) 
    return loss_Data+loss_DataSelf, scores, test_output, labels[np.array(batch_nodes)], labels[test], graphsage, test



  
    

def selfTrain(graphsage, labels,train, test, confList, temperature = 1, update_interval = 75, maxiter = 1000, tol = 2.5, batch_size = 200  ): 
    opt = TrainOptions().parse()
    #index = 0
    #index_array = np.arange(test.shape[0])
    
    closs = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr= opt.lr_slf,  momentum = opt.momentum_slf )
    loss_Data = []
    y_pred_last = 0
    encoder_scheduler = StepLR(optimizer,step_size=opt.step_size ,gamma= opt.gamma)
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:

            labels_pse = np.empty((len(labels), 1), dtype=np.int64)
            labels_pse[train] = labels[train]
            test_scores = graphsage(test)
            
            test_scores = test_scores / temperature.unsqueeze(1).expand(test_scores.size(0), test_scores.size(1))
            test_output, _ = torch.max(F.softmax(test_scores,dim = 1), dim = 1)
            y_pred = F.softmax(test_scores, dim = 1).data.numpy().argmax(axis=1)
            
            conf_test_nod = list((test_output > opt.thres).nonzero().data.numpy().squeeze())
            labels_pse[ test[conf_test_nod]] = calPseLb(F.softmax(test_scores, dim = 1))[conf_test_nod]
            train_set = train + list(test[conf_test_nod])
            print(len(np.array(labels[test[conf_test_nod]])))
    

            print('\nIter {}: '.format(ite), end='')
           
            print("Validation ACCU:", accuracy_score( labels[test], y_pred))
                
            # check stop criterion
            if ite >0:
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float) / y_pred.shape[0]
                print('Fraction of documents with label changes: {} %'.format(np.round(delta_label*100, 3)))
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label <= tol/100:
                print('\nFraction: {} % < tol: {} %'.format(np.round(delta_label*100, 3), tol))
                print('Reached tolerance threshold. Stopping training.')
                break
        
        scores = graphsage(train_set)
        scores = scores / temperature
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
    
    
