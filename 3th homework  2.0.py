# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:56:15 2017

@author: delll
"""


import numpy as np
import operator 
def classify(inputPoint,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat= np.tile(inputPoint,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel= labels[ sortedDistIndicies[i] ]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0] [0] 

if __name__ == "__main__" :
    dataset = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B'] 
    X = np.array([1.2, 1.1])  
    Y = np.array([0.1, 0.1])
    k = 3
    labelX = classify(X,dataset,labels,k)
    labelY = classify(Y,dataset,labels,k)
    print("Your input is:", X, "and classified to class: ", labelX)
    print("Your input is:", Y, "and classified to class: ", labelY)
