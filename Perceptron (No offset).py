# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 20:55:08 2021

@author: user
"""

import numpy as np


def perceptron_no_offset(x,y,epochs):
    '''
    x is a feature matrix
    y is a label vector
    Both need to be numpy arrays
    '''
    #Initialize array theta=0
    theta = np.zeros(x.shape[1])
    
    for t in range(epochs):
        for i in range(x.shape[0]):
            #For every feature row vector in the feature matrix
            if y[i] * np.dot(x[i], theta) <= 0:
                '''
                If y does not match the prediction of the linear classifier 
                (product < 0) or the prediction is uncertain (product is 0), 
                update the parameters of the theta vector.
                '''
                #Nudges the theta parameters in the same/opposite direction
                #(determined by y) of the feature row vector. 
                theta += y[i]*x[i]                           
    return (theta)
