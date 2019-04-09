#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:24:29 2019

@author: ififsun
"""

import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def featToPos(features, node):
    pos =  TSNE(n_components=2).fit_transform(features)
    pos = dict(zip(node,pos))
    
    return pos

def plotGraphStrc(node, features, labels, graph,time, name = "" ):
    

        
    Pos = featToPos(features, node)
    node_labels = dict(zip(node, labels))
    
    edges_all = []
    for edges in graph:
        if edges in node:
            for outer in graph[edges]:
                if outer in node:
                    edges_all.append( (edges, outer))
    ISOTIMEFORMAT = '%Y-%m-%d %H:%M'
    G = nx.Graph()
    G.add_nodes_from(node)
    G.add_edges_from(edges_all)
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.set_title(name)
    nx.draw(G, pos = Pos, label = node_labels,font_size=2, node_color=labels, cmap = plt.cm.Set1, node_size=15)
    f1.savefig("result/Graph_" + name + time.strftime(ISOTIMEFORMAT)+".png", format="PNG")
