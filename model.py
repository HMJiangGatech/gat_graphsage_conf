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
from plotModel import plotGraphStrc
from actL import activeUnsL , activeUnL
"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""


        
def run_cora(device):

    filetime = datetime.datetime.now()
    opt = TrainOptions().parse()
    if opt.dataset  == 'cora':
        opt.k = 80
    elif opt.dataset  =='pubmed':
        opt.k = 50
        opt.lr_pre = 0.04
    elif opt.dataset  == 'ppi':
        opt.lr_pre = 0.03
        opt.k = 40
        opt.epoch = 6000
    elif opt.dataset  == 'reddit':
        opt.lr_pre = 0.06
        opt.k = 80
        opt.epoch = 60000
    writetofile(opt, opt.res_path, filetime)
    np.random.seed(1)
    random.seed(1)
    num_hidden = opt.num_hidden
    feat_data, labels, adj_lists, num_nodes,num_features, train, test, val, num_cls  = load_data(opt.dataset, filetime)
    print(num_cls)
    writetofile('training sample size:'+str(len(train)), opt.res_path, filetime)
    features = nn.Embedding(num_nodes, num_features)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    features = features.to(device)
    graphsage = SupervisedGraphSage(features,  adj_lists, num_features, num_hidden, num_cls, device).to(device)
    
    xent = nn.CrossEntropyLoss()
    

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr= opt.lr_pre, momentum = opt.momentum_pre)
    #encoder_scheduler = StepLR(optimizer,step_size=100,gamma=0.8)
    times = []
    loss_Data = []
    confList = Variable(torch.zeros(num_nodes)).to(device) 
    #optimizer_1 = torch.optim.SGD(graphsage.w, lr=0.5
    for batch in range(opt.epoch):
        batch_nodes = train[:40]
        random.shuffle(train)
        #batch_nodes,_ = sampling(train, confList, k = opt.k)
        start_time = time.time()
        optimizer.zero_grad()
        scores = graphsage(batch_nodes, num_sample = 10, gcn = True)
        conf,_ = scores.max(dim = 1)
        confList[batch_nodes] = conf #update confidence
        l_los = xent(scores, Variable(torch.LongTensor(labels[np.array(batch_nodes)]).type( torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)).squeeze())
        
        loss = l_los 
        
        loss.backward(retain_graph=True)
        
        
        optimizer.step()
        #encoder_scheduler.step(batch)
        end_time = time.time()
        times.append(end_time-start_time)
        loss_Data.append(loss.data)
        if batch%100 == 0:
            out_putT = F.softmax(graphsage(test),dim = 1).data.cpu().numpy().argmax(axis=1)
            print ("Validation ACCU:", accuracy_score(labels[test],  out_putT) )
            #print (batch, loss.data[0])
            
            writetofile("Validation ACCU:"+str( accuracy_score(labels[test], out_putT )), opt.res_path, filetime)
        
        
    test_output =  graphsage(test)
    summary(test, labels, test_output.data.cpu().numpy().argmax(axis = 1), num_cls, filetime, output = test_output , outlog = True)
    print ("Validation ACCU:", accuracy_score(labels[test], F.softmax(test_output,dim = 1).data.cpu().numpy().argmax(axis=1)))
    writetofile("Validation ACCU:"+str( accuracy_score(labels[test],  F.softmax(test_output,dim = 1).data.cpu().numpy().argmax(axis=1))), opt.res_path, filetime)
    loss_DataSelf = []
    ece = plotDiagram(val, graphsage, labels[np.array(val)], 10, filetime)
    writetofile("ECE error:"+str(ece), opt.res_path, filetime)
    ##plotGraphStrc(train + list(test),torch.cat( (graphsage(train).data,test_output.data),dim = 0),  Variable(torch.LongTensor(labels[np.array(train + list(test))])).squeeze().data.numpy(), adj_lists, time = filetime, name = 'pre-train model' )
    #rig, wrg = sepRgtWrg(F.softmax(test_output,dim = 1).data.numpy().argmax(axis=1), labels[np.array( list(test))])
    ##plotGraphStrc(train + list(test[rig])+list(test[wrg]),torch.cat( (graphsage(train).data ,test_output.data[rig], test_output.data[wrg]), dim = 0), [0]*len(train)+[1]*len(rig)+[2]*len(wrg), adj_lists, time = filetime, name = 'pre-train model miscls' )
     
    #model = ModelWithTemperature(graphsage)
    #temperature = model.set_temperature(val, Variable(torch.LongTensor(labels[np.array(val)])).squeeze())
    #temperature = torch.FloatTensor([1])
    #graphsage, loss_DataSelf = selfTrain(graphsage, labels,train, test, val, confList, adj_lists, filetime, features)
    
    
    
    ##test_output = graphsage(test)
    ##test_output = test_output#/temperature.unsqueeze(1).expand(test_output.size(0), test_output.size(1)) 
    
    ##summary(test, labels, test_output.data.numpy().argmax(axis = 1), num_cls, filetime, output = test_output.data.numpy(), outlog = True)
    ##print ("Validation ACCU:", accuracy_score(labels[test], F.softmax(test_output,dim = 1).data.numpy().argmax(axis=1))) 
    ##writetofile("Validation ACCU:"+str( accuracy_score(labels[test], F.softmax(test_output,dim = 1).data.numpy().argmax(axis=1))), opt.res_path, filetime)
    ##plotGraphStrc( list(test),test_output.data,  Variable(torch.LongTensor(labels[np.array( list(test))])).squeeze().data.numpy(), adj_lists, time = filetime, name = 'after self-train model' )
    #plotGraphStrc(train + list(test),torch.cat( (graphsage(train).data,test_output.data),dim = 0),  Variable(torch.LongTensor(labels[np.array(train + list(test))])).squeeze().data.numpy(), adj_lists, time = filetime, name = 'after self-train model' )
    #plotGraphStrc(list(test),test_output.data,  Variable(torch.LongTensor(labels[np.array( list(test))])).squeeze().data.numpy(), adj_lists, time = filetime, name = 'after self-train model' )
    ##plotGraphStrc(list(test),test_output.data, F.softmax(test_output,dim = 1).data.numpy().argmax(axis=1) , adj_lists, time = filetime, name = 'after self-train model classification situcation' )
    
    ##rig, wrg = sepRgtWrg(F.softmax(test_output,dim = 1).data.numpy().argmax(axis=1), labels[np.array( list(test))])
    ##plotGraphStrc(train + list(test[rig])+list(test[wrg]),torch.cat( (graphsage(train).data ,test_output.data[rig], test_output.data[wrg]), dim = 0),[0]*len(train)+[1]*len(rig)+[2]*len(wrg) , adj_lists, time = filetime, name = 'after self-train model miscls' )
    return loss_Data+loss_DataSelf, scores, test_output, labels[np.array(batch_nodes)], labels[test], graphsage, test, filetime



  
    

def selfTrain(graphsage, labels, train, test, val, confList, adj_lists, filetime, features, temperature = 1, update_interval = 50, maxiter = 500, tol = 0.5, batch_size = 200  ): 
    opt = TrainOptions().parse()
    #index = 0
    #index_array = np.arange(test.shape[0])
    
    closs = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr= opt.lr_slf,  momentum = opt.momentum_slf )
    loss_Data = []
    y_pred_last = 0
    encoder_scheduler = StepLR(optimizer,step_size=opt.step_size ,gamma= opt.gamma)
    acNodes = []
    nonActNod  = []
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:

            labels_pse = np.empty((len(labels), 1), dtype=np.int64)
            labels_pse[train] = labels[train]
            test_scores = graphsage(test)
            
            test_scores = test_scores #/ temperature.unsqueeze(1).expand(test_scores.size(0), test_scores.size(1))
            test_output, _ = torch.max(F.softmax(test_scores,dim = 1), dim = 1)
            y_pred = F.softmax(test_scores, dim = 1).data.numpy().argmax(axis=1)
            
            #conf_test_nod, unconf_test_nod = vat_sel(graphsage, test, test_scores)
            conf_test_nod = list((test_output > opt.thres).nonzero().data.numpy().squeeze())
            unconf_test_nod = list((test_output <= opt.thres).nonzero().data.numpy().squeeze())
            #print(len(list((test_output <= 0.4).nonzero().data.numpy().squeeze())))
            labels_pse[ test[conf_test_nod]] = calPseLb(F.softmax(test_scores, dim = 1))[conf_test_nod]
            
            
            if ite % (update_interval*2) == 0:
                node, nonActN =  activeUnL(graphsage, list(val), Variable(torch.LongTensor(labels[np.array(list(val))])).squeeze().data.numpy(), features, adj_lists, opt.num_features, opt.num_hidden, opt.num_cls, filetime, labels)
                acNodes.extend(node)
                nonActNod.extend(nonActN)
                train_set = train +list(acNodes)
                labels_pse[ acNodes ] = labels[ acNodes ]
                #print(len(conf_test_nod), len(actNod))
                #writetofile(len(actNod), opt.res_path, filetime)
                #plotGraphStrc(train + acNodes + nonActNod,graphsage(train + acNodes + nonActNod).data, [0]*len(train)+[1]*len(acNodes)+[2]*len(nonActNod), adj_lists, time = filetime, name = 'act-slfT'+str(ite // update_interval))
            else:
                train_set = train + list(test[conf_test_nod])+list(acNodes)
                labels_pse[ acNodes ] = labels[ acNodes ]
            
            
            
            writetofile(len(conf_test_nod), opt.res_path, filetime)
            
            plotGraphStrc(train + list(test[conf_test_nod]) + list(test[unconf_test_nod]),graphsage(train + list(test[conf_test_nod]) + list(test[unconf_test_nod])).data, [0]*len(train)+[1]*len(conf_test_nod)+[2]*len(unconf_test_nod), adj_lists, time = filetime, name = 'slfT'+str(ite // update_interval))
            #plotGraphStrc(train + list(test) ,graphsage(train + list(test)).data, [0]*len(train)+ list(test_output) , adj_lists, time = filetime, name = 'slfT'+str(ite // update_interval))
            print('\nIter {}: '.format(ite), end='')
            writetofile('Iter:'+str(ite), opt.res_path, filetime)
            print("Validation ACCU:", accuracy_score( labels[test], y_pred))
            writetofile("Validation ACCU:" +  str(accuracy_score( labels[test], y_pred)), opt.res_path, filetime)
            print("confidence point ACCU:", accuracy_score( labels[test[conf_test_nod]], y_pred[conf_test_nod]))
            writetofile("confidence point ACCU:" +  str(accuracy_score( labels[test[conf_test_nod]], y_pred[conf_test_nod])), opt.res_path, filetime)
            # check stop criterion
            if ite >0:
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float) / y_pred.shape[0]
                print('Fraction of documents with label changes: {} %'.format(np.round(delta_label*100, 3)))
                writetofile("Fraction of documents with label changes: " + str(np.round(delta_label*100, 3)), opt.res_path, filetime)
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label <= tol/100:
                print('\nFraction: {} % < tol: {} %'.format(np.round(delta_label*100, 3), tol))
                print('Reached tolerance threshold. Stopping training.')
                writetofile('Reached tolerance threshold. Stopping training.', opt.res_path, filetime)
                break
            
        scores = graphsage(train_set)
        #scores = scores #/ temperature
        #v_loss = vat_loss(graphsage, train_set,scores)
        loss = closs(scores, Variable(torch.LongTensor(labels_pse[np.array(train_set)])).squeeze()) #+  activeL(graphsage, train_set, Variable(torch.LongTensor(labels_pse[np.array(train_set)])).squeeze(), features, adj_lists, opt.num_features, opt.num_hidden, opt.num_cls)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        encoder_scheduler.step(ite)
        #index = index + 1 if (index + 1) * batch_size <= test.shape[0] else 0   
        loss_Data.append(loss.data)
    return graphsage, loss_Data

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
            
    loss_Data, scores, val_output, labels_train, labels_val, graphsage, test, filetime = run_cora(device) #0.862 time 0.0506  #0.888 - 0.894 avg time 0.74 # 0.846
    ISOTIMEFORMAT = '%Y-%m-%d %H:%M'
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    plt.plot(range(len(loss_Data)), loss_Data)
    plt.draw()
    plt.show()
    f1.savefig("result/lossData"+filetime.strftime(ISOTIMEFORMAT)+".png", format="PNG")

    #loss_Data_p = run_pubmed() #0.808  time: 0.4415 #0.832  avg time 15.62  #80.2    
    #plt.plot(range(len(loss_Data_p)), loss_Data_p) 
    
    

