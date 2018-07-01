#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 12:38:20 2018

@author: TheMrGhostman
"""
import NeuralNetwork as NN
from time import time
#import numpy as np

#s = time()
brain = NN.NeuralNetwork(2,2,1)

inputs = [[1, 0], [1, 1], [0,1], [0,0]]
res = [1, 0, 1 , 0]

#inputs = [1, 0]
#res = 1

output = brain.predict(inputs)
#e = time()
print(output)
#print(e-s)

brain.fit(inputs, res)

print(brain.predict([1,0]))