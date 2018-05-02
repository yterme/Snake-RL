#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:41:48 2018

@author: yannickterme
"""

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import numpy as np


class DQNetwork:
    def __init__(self, actions, input_shape, alpha=0.1, gamma=0.9,
                 dropout_prob=0.1, load_path='', conv = False, logger=None):
        self.input_shape = input_shape
        self.model = Sequential()
        self.actions = actions  # Size of the network output
        self.gamma = gamma
        self.alpha = alpha
        self.dropout_prob = dropout_prob

        # Define neural network
        
        if conv:
            self.model.add(BatchNormalization(axis=1, input_shape=input_shape))
            self.model.add(Convolution2D(32, 2, 2, border_mode='valid',
                                         subsample=(2, 2), dim_ordering='th'))
            self.model.add(Activation('relu'))
    
            self.model.add(BatchNormalization(axis=1))
            self.model.add(Convolution2D(64, 2, 2, border_mode='valid',
                                         subsample=(2, 2), dim_ordering='th'))
            self.model.add(Activation('relu'))
    
            self.model.add(BatchNormalization(axis=1))
            self.model.add(Convolution2D(64, 3, 3, border_mode='valid',
                                         subsample=(2, 2), dim_ordering='th'))
            self.model.add(Activation('relu'))
    
            self.model.add(Flatten())
    
            self.model.add(Dropout(self.dropout_prob))
            
            
            self.model.add(Dense(256))
            self.model.add(Activation('relu'))
            
            self.model.add(Dense(self.actions))
    
            self.optimizer = Adam()
            
        else:
            self.model.add(BatchNormalization(axis=1, input_shape=input_shape))
            self.model.add(Flatten())
            
            self.model.add(Dense(512))
            self.model.add(Activation('relu'))
            
            self.model.add(Dropout(self.dropout_prob))
            
            self.model.add(Dense(256))
            self.model.add(Activation('relu'))

            self.model.add(Dense(128))
            self.model.add(Activation('relu'))
            
            self.model.add(Dense(self.actions))
            
            self.optimizer = Adam()
            
        # Load the network from saved model
        self.logger = logger
        
        if load_path != '':
            self.load(load_path)

        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer,
                           metrics=['accuracy'])

    def train(self, batch):
        """
        Generates inputs and targets from the given batch, trains the model on
        them.
        :param batch: iterable of dictionaries with keys 'source', 'action',
        'dest', 'reward'
        """
        x_train = []
        t_train = []

        # Generate training set and targets
        for datapoint in batch:
            x_train.append(datapoint['source'].astype(np.float64))

            # Get the current Q-values for the next state and select the best
            next_state_pred = self.predict(datapoint['dest'].astype(np.float64)).ravel()
            next_q_value = np.max(next_state_pred)

            # The error must be 0 on all actions except the one taken
            t = list(self.predict(datapoint['source'])[0])
            if datapoint['final']:
                t[datapoint['action']] = datapoint['reward']
            else:
                t[datapoint['action']] = datapoint['reward'] + \
                                         self.gamma * next_q_value
            t_train.append(t)

        # Prepare inputs and targets
        x_train = np.asarray(x_train).squeeze()
        t_train = np.asarray(t_train).squeeze()

        # Train the model for one epoch
        h = self.model.fit(x_train.reshape(tuple([len(x_train)]+list(self.input_shape))),
                           t_train,
                           batch_size=32,
                           nb_epoch=1)

        # Log loss and accuracy
        if self.logger is not None:
            self.logger.to_csv('loss_history.csv',
                               [h.history['loss'][0], h.history['acc'][0]])

    def predict(self, state):
        """
        Feeds state into the model, returns predicted Q-values.
        :param state: a numpy.array with same shape as the network's input
        :return: numpy.array with predicted Q-values
        """
        state = state.astype(np.float64)
        return self.model.predict(state, batch_size=1)

    def save(self, filename=None):
        """
        Saves the model weights to disk.
        :param filename: file to which save the weights (must end with ".h5")
        """
        f = ('model.h5' if filename is None else filename)
        if self.logger is not None:
            self.logger.log('Saving model as %s' % f)
        self.model.save_weights(self.logger.path + f)

    def load(self, path):
        """
        Loads the model's weights from path.
        :param path: h5 file from which to load teh weights
        """
        if self.logger is not None:
            self.logger.log('Loading weights from file...')
        self.model.load_weights(path)