#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:45:53 2019

@author: ififsun
"""
scores = scores.data.numpy()
val_output = val_output.data.numpy() 


scores = np.array(scores)
val_output = np.array(val_output)


scores_1 = np.max(scores, axis = 1)
val_output_1 = np.max(val_output, axis = 1)
pred_t = np.argmax(scores, axis = 1)
pred_v = np.argmax(val_output, axis = 1)


acc = np.zeros(10)
conf = np.zeros(10)
cont = np.zeros(10)
for j in range(len(scores_1)):
    for i in [i/10.0 for i in range(10)]:
        if scores_1[j] >= i and scores_1[j]< (i+0.1):
            cont[int(i*10)]+=1
            acc[int(i*10)] += 1 if pred_t[j] == labels_train[j][0] else 0
            conf[int(i*10)] += scores_1[j]
np.sum((acc - conf)/80)
        

acc = np.zeros(10)
conf = np.zeros(10)
cont = np.zeros(10)
for j in range(len(val_output_1)):
    for i in [i/10.0 for i in range(10)]:
        if val_output_1[j] >= i and val_output_1[j]< (i+0.1):
            cont[int(i*10)]+=1
            acc[int(i*10)] += 1 if pred_v[j] == labels_val[j][0] else 0
            conf[int(i*10)] += val_output_1[j]

np.sum(abs(acc - conf)/80)