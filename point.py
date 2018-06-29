#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 16:42:04 2018

@author: root
"""
from random import uniform

class Point:
    def __init__(self):
        self.x = uniform(0,200)
        self.y = uniform(0,200)
        if self.x > self.y:
            self.label = 1
        else:
            self.label = -1
            
    def println(self):
        #print(self.x, self.y, self.label)
        return(self.x, self.y, self.label)