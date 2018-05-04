#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 14:26:48 2018

@author: yannickterme
"""

import argparse
import os

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
        
    def initialize(self):
        self.parser.add_argument('--name', required=True, help='experience name')
        self.parser.add_argument('--gridsize', type=int, default=5, help='size of grid (nrow=ncol)')
        self.parser.add_argument('--n_train', type=int, default = 1000, help= 'nb of training rounds')
        self.parser.add_argument('--conv', action = 'store_true', help='if specified, use a CNN ')
        self.parser.add_argument('--load', action='store_true', help= 'if specified, load model')
        self.parser.add_argument('--n_channels', type = int, default= 1,help='number of channels of the image')
        self.parser.add_argument('--n_episodes', type=int, default = 100, help='number of episodes per training round')
        self.initialized = True
    
    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        args = vars(opt)
        self.opt = opt
        return self.opt