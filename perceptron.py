#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 16:06:25 2018

@author: Ghostman
"""
import numpy as np

def add_bias(inputs, TFbias):
    if TFbias:
        return np.hstack((np.array(inputs), np.ones(1)))
    else:
        return np.array(inputs)
    
class Perceptron:    
    
    def __init__(self, n_features, addBias = True):
        #adding bias or not
        self.TFbias = addBias
        #initialize wights randomly 
        self.weights = np.random.uniform(-1, 1, n_features + self.TFbias)
        self.learning_rate = 0.1
        self.error = 'nan'
    
    def predict(self, inputs):
        return np.sign(np.dot(add_bias(inputs, self.TFbias), self.weights))
    
    def fit(self, inputs, label):
        if isinstance(label, list):
            for i, point in enumerate(inputs):
                guess = Perceptron.predict(self, point)
                self.error = label[i] - guess
                self.weights = self.weights + self.error*\
                                add_bias(point, self.TFbias)*self.learning_rate
        else:
            guess = Perceptron.predict(self, inputs)
            self.error = label - guess
            self.weights = self.weights + self.error*\
                           add_bias(inputs, self.TFbias)*self.learning_rate
        
    def score(self):
        return(self.error)
        
"""        
p = Perceptron()    
inp = [-1,0.5]
print(p.predict(inp))
"""