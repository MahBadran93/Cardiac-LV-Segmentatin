#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 06:29:37 2020

@author: tarek
"""


# -*- coding: utf-8 -*-
import cv2
import  numpy as np
import math
def set_clicked_center(img):
    '''
    Show image and register the coordinates of a click into
    a global variable.
    '''
    def detect_click(event, x, y, flags, param):
        global click
        click = (x, y)

    cv2.namedWindow("clicked", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("clicked", detect_click)

    while True:
        cv2.imshow("clicked", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if click:
            return click
        
def initalizeShape(centroid,meanShapeCentroid, modelShape):
    # centroid : center coords , i.e : (120,120).
    # modelShape : the shape to center around centroid. 
    subtractValue = np.subtract(np.array(centroid),np.array(meanShapeCentroid))
    translatedShpe = np.add(modelShape,subtractValue)
    
    return translatedShpe # return a translated shape around a specific fixed centroid for all shapes 
    
'''
def transformShape(pose_para, b):
     
    #Deform the shape to new b(eigenvalue) parameters and a new pose parameters 
     
     # x = x_mean +bp , deform
     # proc. analysis to get the new pose parameters 
     mode = self.pdmodel.deform(b)
     # here add instead the proc. analysis 
     return self.aligner.transform(mode, pose_para)
'''
def translateShape(shape, Tx, Ty):
    '''
    Translate a shape according to translation parameters
    '''
    shape[:,0] += Tx # add Tx to the x axis values 
    shape[:,1] += Ty # add Ty to the y axis values
    return shape  # return translated shape 

def rotateShape(translatedShape, s, theta, inverse=False):
    '''Rotate over theta and scale by s'''
    rotation_matrix = np.array([
                        [s*math.cos(theta), -1*s*math.sin(theta)],
                        [s*math.sin(theta), s*math.cos(theta)]
                        ])
    for i in range(len(translatedShape)):
        translatedShape[i] = np.dot(rotation_matrix, translatedShape[i].T)
    
    # if inverse:
    #     return np.dot(rotation_matrix.T, translatedShape)
    # else:
    #     return np.dot(rotation_matrix, translatedShape)
    return translatedShape
    
def getNorm():
    pass

arr = np.zeros((30,2))

ss = translateShape(arr, 1,0)
gg = rotateShape(ss,1, math.radians(360))