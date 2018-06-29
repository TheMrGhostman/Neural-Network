#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 16:06:25 2018

@author: Ghostman
"""
import numpy as np

class Perceptron:    
    
    def __init__(self):
        #initialize wights randomly 
        self.weights = np.random.uniform(-1, 1, 2)    
        self.learning_rate = 0.1
        
    def predict(self, inputs):
        return np.sign(np.dot(np.array(inputs), self.weights))
    
    def fit(self, inputs, label):
        if isinstance(label, list):
            for i, point in enumerate(inputs):
                guess = Perceptron.predict(self, point)
                error = label[i] - guess
                self.weights = self.weights + error*np.array(point)*self.learning_rate
        else:
            guess = Perceptron.predict(self, inputs)
            error = label - guess
            self.weights = self.weights + error*np.array(inputs)*self.learning_rate
        
        
        
"""        
p = Perceptron()    
inp = [-1,0.5]
print(p.predict(inp))
"""