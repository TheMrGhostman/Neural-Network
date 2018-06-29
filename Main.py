#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 16:50:14 2018

@author: root
"""

import perceptron 
import point as p
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle


body = []

plt.figure("test")
color = ["red", "green", "blue"]
plt.plot(np.arange(0,200),np.arange(0,200), color = "red")
for i in range(200):
    n = p.Point()
    m = p.Point.println(n)
    plt.scatter(m[0], m[1] , color = color[m[2]])
    body.append(n)
plt.show()


brain = perceptron.Perceptron()

for j in range(10):
    shuffle(body)
    for pt in body:
        inputs = [pt.x, pt.y]
        brain.fit(inputs, pt.label)


plt.figure("testing")
plt.plot(np.arange(0,200),np.arange(0,200), color = "red")
for i in range(1000):
    b = p.Point()
    plt.scatter(b.x, b.y, color = color[int(brain.predict([b.x, b.y]))])
plt.show()