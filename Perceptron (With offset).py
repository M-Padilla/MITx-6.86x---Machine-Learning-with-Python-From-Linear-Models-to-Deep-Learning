# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 23:54:35 2021

@author: M-Padilla
"""

import numpy as np


def perceptron(x,y,epochs):
    '''
    x is a feature matrix
    y is a label vector
    Both need to be numpy arrays
    '''  
    #Initialize array theta=0
    theta = np.zeros(x.shape[1]+1) #theta[-1] is the offset, theta[:-1] are the parameters
    
    for t in range(epochs):
        for i in range(x.shape[0]):
            #For every feature row vector in the feature matrix            
            if y[i] * (np.dot(x[i], theta[:-1]) + theta[2]) <= 0:
                '''
                If y does not match the prediction of the linear classifier 
                (product < 0) or the prediction is uncertain (product is 0), 
                update the parameters of the theta vector.
                '''
                #Nudges the theta parameters in the same/opposite direction
                #(determined by the label y[i]) of the feature row vector.
                theta[:-1] += y[i]*x[i]
                #Increases/decreases the offset (determined by the label y[i]).
                theta[-1] += y[i]                                   
    
    return theta       
  
