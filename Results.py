#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:50:28 2018

@author: yannickterme
"""

import numpy as np

class Results():
    def __init__(self, lengths=[], scores =[], lengths_expl=[],scores_expl=[], \
                 memory = [], epsilon = 1):
        self.lengths = lengths
        self.scores = scores
        self.lengths_expl = lengths_expl
        self.scores_expl = scores_expl
        self.epsilon = epsilon
        self.memory = memory
        

def test(env, model, n_channels, episode_count = 100, imax = 100):
    lengths = []
    scores = []
    for i_episode in range(episode_count):
        #print("Episode",str(i_episode))
        grid = env.reset()
        i=0
        done = False
        epsilon = 0

        while i <imax:
            grid= grid.reshape((1,n_channels, env.nrow, env.ncol))
            action = np.argmax(model.predict(grid))
            grid, reward, done = env.step(action)

            i+=1
            if done:
                break
        lengths.append(i)
        scores.append(env.score)
    mean_length = np.mean(lengths)
    mean_score = np.mean(scores)

    return(mean_length, mean_score)