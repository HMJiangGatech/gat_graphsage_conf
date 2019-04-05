#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:02:29 2019

@author: ififsun
"""

import argparse


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, default='./data', help='path')
        self.parser.add_argument('--dataset', type=str, default='pubmed', help='[cora | citeseer | pubmed|ppi]')
        self.parser.add_argument('--num_features', type=int, default=1433, help='number of features')
        self.parser.add_argument('--num_hidden', type=int, default=180, help='number of hidden layer dimension')
        self.parser.add_argument('--res_path',type=str, default= 'result/result_para',help = 'the path to save result and parameter')

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain
        args = vars(self.opt)

        return self.opt

class TrainOptions(BaseOptions):
    # Override
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--epoch',type=int, default= 150,help = 'number of epoch')
        self.parser.add_argument('--k',type=int, default= 80,help = 'select the least k sample to train during the pre-train')
        self.parser.add_argument('--lr_pre',type=int, default= 0.02,help = 'learning rate in pre-train')
        self.parser.add_argument('--momentum_pre',type=int, default= 0.9,help = 'momentum in pre-train')
        self.parser.add_argument('--lr_slf',type=int, default= 0.03,help = 'learning rate in self-train')
        self.parser.add_argument('--momentum_slf',type=int, default= 0.9,help = 'momentum in self-train')
        self.parser.add_argument('--step_size',type=int, default= 100,help = 'step size for learning rate decay in self-train')
        self.parser.add_argument('--gamma',type=int, default= 0.8,help = 'gamma for learning rate decay in self-train')
        self.parser.add_argument('--thres',type=int, default= 0.995,help = 'threshold for self-training pseudo data selection')
        self.isTrain = True