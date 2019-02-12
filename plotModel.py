#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:24:29 2019

@author: ififsun
"""

import networkx as nx
G = nx.Graph()
G.add_nodes_from([1, 2, 3,4,5])
G.add_edges_from([(1, 2,{'weight': 0.1}), (1, 3,{'weight': 10}), (2, 3,{'weight': 9}), (3, 4,{'weight': 5}), (3, 5,{'weight': 2}), (1, 5,{'weight': 1}), (1, 4,{'weight': 0.5})])

import matplotlib.pyplot as plt
nx.draw_kamada_kawai(G, with_labels=True, font_weight='bold')


def plotGraphStrc(features, labels, graph ):
    
    
nx.draw_networkx_labels(G,[[1,2,3],[5,4,3], [2,5,1], [2,3,5], [6,4,2]],labels,font_size=16)
