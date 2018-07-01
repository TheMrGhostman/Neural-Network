#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 12:35:47 2018

@author: TheMrGhostman
"""
import numpy as np

def sigmoid(X):        
    return 1/(1 + np.exp(-X))

def dsigmoid(Y):
    out = np.multiply(Y,(1 - Y)) 
    #print(out)
    return out
    
class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs, activation="sigmoid", learning_rate = 0.1):
        self.input_nodes = n_inputs
        self.hidden_nodes = n_hidden
        self.output_nodes = n_outputs
        self.activation = activation
        self.learning_rate = learning_rate
        
        # IH - Inputs to Hidden
        self.weights_IH = np.random.uniform(-1, 1, n_inputs*n_hidden).\
                                reshape(n_hidden, n_inputs)
        self.bias_IH =  np.matrix(np.random.uniform(-1, 1, n_hidden).\
                                reshape(n_hidden, 1))
        # HO - Hidden to Output
        self.weights_HO = np.random.uniform(-1, 1, n_hidden*n_outputs).\
                                reshape(n_outputs, n_hidden)
        self.bias_HO =  np.matrix(np.random.uniform(-1, 1, n_outputs).\
                                reshape(n_outputs, 1))
                                
        
    def predict(self, inputs):
        """
        feedforward algorithm
        """
        #transpozice vstupu
        inputs = np.matrix(inputs).T
        
        hidden = np.matmul(self.weights_IH, inputs) + self.bias_IH
        hidden = sigmoid(hidden)
        
        output = np.matmul(self.weights_HO, hidden) + self.bias_HO
        output = sigmoid(output)
        
        return np.squeeze(np.asarray(output))
    
    
    def fit(self, inputs, results):
        """
        backpropagation algorithm
        """
        #transpozice vstupu
        inputs = np.matrix(inputs).T
        print(inputs)
        
        hidden = np.matmul(self.weights_IH, inputs) + self.bias_IH
        hidden = sigmoid(hidden)
        
        output = np.matmul(self.weights_HO, hidden) + self.bias_HO
        output = sigmoid(output)
        print("output", output)
        #output = np.matrix(output).T
        results = np.matrix(results).T
        
        output_errors = results - output.T  
        print(output_errors)
        
        grad_HO = np.multiply(np.matmul(dsigmoid(output), output_errors), self.learning_rate)
        print("grad HO" , grad_HO)
        dW_HO = grad_HO * hidden.T
        print("dwho", dW_HO)
        self.weights_HO += dW_HO
        #self.bias_HO += grad_HO
        print("bias", self.bias_HO)
        
        hidden_errors = np.matmul(self.weights_IH.T, output_errors)
        grad_IH = dsigmoid(hidden) * hidden_errors * self.learning_rate
        print("grad IH", grad_IH)
        dW_IH = grad_IH * inputs.T
        self.weights_IH += dW_IH
        #self.bias_IH += grad_IH
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    