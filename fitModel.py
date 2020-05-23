#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon april 28 07:44:14 2020

@author: tarek
"""

import numpy as np
import cv2


from dentalvision.utils.structure import Shape

import matplotlib.pyplot as plt
import  sklearn.metrics as sk
import random
import scipy
import math
from AutomateLandMarks import LandMarks
from dentalvision.glm.profile import Profiler

from skimage import exposure
from skimage import feature, img_as_float

from dentalvision.utils.align import Aligner
from dentalvision.utils.align import CoreAlign





profiler = Profiler()
proc= LandMarks()
aligner= Aligner()
pose_param=[]
core= CoreAlign()

def calc_external_img_active_contour(img): 

  #  median = median_filter(img)
    contrast = contrast_stretching(img)
    ext_img = canny(contrast)

    return ext_img

def median_filter(img, size = 3):
    return scipy.signal.medfilt(img, size).astype(np.uint8)

def contrast_stretching(img):
    # Contrast stretching
    #p2, p98 = np.percentile(img, (0, 20))
    return exposure.rescale_intensity(img, in_range=(0.05*255, 0.6*255))

def canny(img):
    return feature.canny(img, sigma=2)


def preperation(radiograph):

    #median = prep.median_filter(radiograph)
#     edge_img = prep.edge_detection_low(median)

    edge_img = calc_external_img_active_contour(radiograph)
    
    return edge_img   


def fit_measure(points, length, edge_img):
    "point: touth point  initial shape "
    
  
    size = len(points)
    # print("point is ",points)
    new_points = np.empty((size,2))
    total_error = 0
    
    
    for i in range(size):
        #print(i)
        if(i==size-1):
            p1, p2, p3 = points[i-1], points[i], points[0] 
        else:
            p1, p2, p3 = points[i-1], points[i], points[i+1]
            
        # plt.plot(points[:,0],points[:,1])
        p2_new = strongest_edge_point_on_normal(p1, p2, p3 ,length, edge_img)
        # plt.imshow(edge_img)
        # plt.plot(p2_new[0],p2_new[1],".r")
        # plt.show()
        
        total_error += error_measure(p2, p2_new)
        new_points[i] = p2_new
      
    
       # print(new_points.shape,new_points[1])
    # plt.plot(new_points[:,0],new_points[:,1],".")
    # plt.show()    
    return new_points, total_error;  

# def strongest_edge_point_on_normal(a,b,c,length, edge_img):
    
    
#     rad = get_normal_angle(a,b,c)
#     points = get_points_on_angle(b, rad, length)
#     edge_strength = edge_strength_at_points(points, edge_img)
#     id_edge_point = np.argmax(edge_strength)
#     edge_point =s points[id_edge_point]

#     return edge_point 
    

def strongest_edge_point_on_normal(a,b,c,length, edge_img):
    
    profiler.reset(length)
    points = profiler.sample_1(a,b,c)
    # plt.imshow(edge_img)
    edge_strength = edge_strength_at_points(points, edge_img)
    # plt.plot(edge_strength[:,0],edge_strength[:,1])
    id_edge_point = np.argmax(edge_strength)
    edge_point = points[id_edge_point]
    
    # print(edge_point)
    # plt.imshow(edge_img)
    # plt.plot(points[:,0],points[:,1])
    # plt.plot(edge_point[0],edge_point[1],'.r')
    # plt.show()


    
    

    return edge_point 


def edge_strength_at_points(points ,edge_img):
    
    gradient = np.empty(len(points))
    for i, p in enumerate(points):
        gradient[i] = edge_img[int(p[1]),int(p[0])]
        
    return gradient


def error_measure(p1, p2):
    
    x1, y1 = p1
    x2, y2 = p2
    #dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return math.hypot(x2 - x1, y2 - y1)



def get_range_of(i, pca):
    eigenvalues = pca.eigenvalues
    bound = 3*math.sqrt(eigenvalues[i])
    return bound


def generate_model_point(b,pca):
    eigenvectors = pca.scaled_eigenvectors
    P = eigenvectors
    #b = get_eigenvalues(pca)
    xm = pca.mean.vector
    
    x =  np.dot(P,b)
    return x + xm


def get_pose_param_for_transformation(A,B):
    print ("heeeere ")
    print ("A",A.shape)
    print ("B",B.shape)
    s = round(len(A)/3)
    r = random.randint(0, s) 
    r2 , r3 = round(r+s), round(r+2*s)
    
    pts1 = np.float32((A[r],A[r2],A[r3]))
    pts2 = np.float32((B[r],B[r2],B[r3]))
    M = cv2.getAffineTransform(pts1,pts2)
    
    ax, ay, tx, ty = M[0,0], M[1,0], M[0,2], M[1,2]
    return (ax, ay, tx, ty)

def getparameter(A,B):
      pose_matrix,disparity,pose_param= proc.procrustes(A,B)
      
      return pose_param
    

def inv_transform(p, pose_param):
    

    (ax, ay, tx, ty) = pose_param
    scale_rotation = np.array([[ax, ay],[ -ay, ax]])
    divisor = 1 / (ax**2 + ay**2)
    Txy = np.array([tx, ty])
    T= np.transpose(p-Txy)
    #print (T.shape)
    transformed = np.dot(scale_rotation*divisor, T)
    #print (transformed.shape)
        
    return transformed.transpose()

def transform(p, pose_param):
    (ax, ay, tx, ty) = pose_param
    scale_rotation = np.array([[ax, -ay],[ ay, ax]])
    Txy = np.array([tx, ty])
    
    p_transformed = np.dot(scale_rotation, p.transpose()) 
    return (np.add(Txy, p_transformed.transpose()))

def project_to_tangent_plane(y, pca):
    xm = pca.mean.vector
    y= Shape(y)
    #y = y.reshape(-1)
    #print ("project function ", y.vector.shape)
    return y.vector / np.dot(y.vector,xm)

def update_model_param(y, pca):
    y= Shape(y)
    xm = pca.mean
    PT = pca.eigenvectors.T
    return np.dot(PT, y.vector - xm.vector)# update eigenvalue 

def constraint_model_param(b,pca):
    
    for i in range(len(b)):
        bound = get_range_of(i,pca)
        if( b[i] > bound):
            b[i] = bound
        elif ( b[i] < -bound):
            b[i] = -bound
    return b




def match_model_points(Y, pca):
    b = np.zeros(len(pca.eigenvalues)) 

    max_conv_iter = 20
    best_b = b
    best_pose_param = (0,0,0,0)
    best_MSE = np.inf
    convergence_iter = max_conv_iter

    while(1):
        
        x = generate_model_point(b, pca)
        x= Shape(x)
    
        plt.plot(x.x,x.y,'.')
        plt.show()
        
        pose_param=getparameter(Y,x.matrix.T)
        
        global pose_param2
        pose_param2=pose_param
    
        pose_param= pose_param2['rotation'][0,0],pose_param2['rotation'][0,1],pose_param2['translation'][0],pose_param2['translation'][1]
    
        Y_pred = inv_transform(x.matrix.T,pose_param)
        #print(Y_pred.shape)
        # plt.plot(Y_)
        
        #MSE = sk.mean_squared_error(Y.matrix.T, Y_pred)
        MSE = sk.mean_squared_error(Y, Y_pred)

        
        if(MSE < best_MSE):
            best_b = b
            best_pose_param = pose_param
            best_MSE = MSE
            convergence_iter = max_conv_iter
        
        convergence_iter -= 1
        if(convergence_iter == 0 or best_MSE < 1):
            # print(convergence_iter, best_MSE)
            break;    
          
        #y = transform(Y.matrix.T,pose_param)
        y = transform(Y,pose_param)
        y = project_to_tangent_plane(y, pca)
        
        # x,y = np.split(np.hstack(y), 2)
        # y = np.vstack((x, y))
        
        # print (y.shape)
        # y = np.vstack((y[0:29], y[30:59])).transpose()
        y=Shape(y)
        b = update_model_param(y.vector, pca)
      #  print(b)

        b = constraint_model_param(b,pca)
    
    
    return best_b, best_pose_param;



# def match_model_points(Y, pca):
#     b = np.zeros(len(pca.eigenvalues)) 
#     aligner= Aligner()
#     fitter=Fitter(pca)  

#     max_conv_iter = 20
#     best_b = b
#     best_pose_param = (0,0,0,0)
#     best_MSE = np.inf
#     convergence_iter = max_conv_iter

#     while(1):
        
#         x = generate_model_point(b, pca)
#         x = Shape(x)
#         plt.plot(x.x,x.y)
#         plt.show()
        
          
#         pose_param = aligner.get_pose_parameters(Y,x)
        
#         # set initial fitting pose parameters
    
#         # align model mean with region
#         # pose_param = get_pose_param_for_transformation(Y,x.matrix.T)
        
#         Y_pred = aligner.invert_transform(x,pose_param)
#         MSE = sk.mean_squared_error(Y.matrix, Y_pred.matrix)
        
#         if(MSE < best_MSE):
#             best_b = b
#             best_pose_param = pose_param
#             best_MSE = MSE
#             convergence_iter = max_conv_iter
        
#         convergence_iter -= 1
#         if(convergence_iter == 0 or best_MSE < 1):
#             #print(convergence_iter, best_MSE)
#             break;    
         
#         y = aligner.transform(Y,pose_param)
        
        
#         y = project_to_tangent_plane(y, pca)
        
#         y = Shape(y)

#         b = update_model_param(y.matrix.T, pca)
#         b = constraint_model_param(b,pca)

#     return best_b, best_pose_param;

# # def match_model_points(Y, pca):
# #     b = np.zeros(len(pca.eigenvalues)) 

# #     max_conv_iter = 20
# #     best_b = b
# #     best_pose_param = (0,0,0,0)
# #     best_MSE = np.inf
# #     convergence_iter = max_conv_iter

# #     while(1):
        
# #         x = generate_model_point(b, pca)
        
# #         x= Shape(x)
        
# #         Y=Shape(Y)
            
    
# #         # plt.plot(x.x,x.y,'.')
# #         # plt.show()


# #         fitter=Fitter(pca)
# #         fitter.start_pose=aligner.get_pose_parameters(Y, x)
# #         global pose_param
# #         pose_para, c = fitter.fit(Y, x)
# #         print(pose_para)
# #         print (c)
# #         # pose_param1=tarekfunctionforposeparameter(Y.matrix.T,x.matrix.T)
        
# #         # global pose_param2
# #         # pose_param2=pose_param1
    
# #         # pose_param= pose_param2['rotation'][0,0],pose_param2['rotation'][0,1],pose_param2['translation'][0],pose_param2['translation'][1]
        
        
     
# #         Y_pred = inv_transform(x.matrix.T,pose_para)
# #         #print(Y_pred.shape)
# #         # plt.plot(Y_)
        
# #         #MSE = sk.mean_squared_error(Y.matrix.T, Y_pred)
# #         MSE = sk.mean_squared_error(Y.matrix.T, Y_pred)

        
# #         if(MSE < best_MSE):
# #             best_b = b
# #             best_pose_param = pose_para
# #             best_MSE = MSE
# #             convergence_iter = max_conv_iter
        
# #         convergence_iter -= 1
# #         if(convergence_iter == 0 or best_MSE < 1):
# #             # print(convergence_iter, best_MSE)
# #             break;    
          
# #         #y = transform(Y.matrix.T,pose_param)
# #         y = transform(Y.matrix.T,pose_param)
# #         y = project_to_tangent_plane(y, pca)
        
# #         # x,y = np.split(np.hstack(y), 2)
# #         # y = np.vstack((x, y))
        
# #         # print (y.shape)
# #         # y = np.vstack((y[0:29], y[30:59])).transpose()
# #         y=Shape(y)
# #         b1 = update_model_param(y.vector, pca)
# #       #  print(b)

# #         b = constraint_model_param(b1,pca)
# # #       #  print (b )
# # #         if b1.all==b.all:
# # #             print ("yes")
# # #         else:
# # #             print ("no")

# # # #     plt.plot(show_var.x,show_var.y,'.')
        
        
    

#     return best_b, best_pose_param;


def show_evolution(img, points_list):
    
    plt.figure()
    fig, ax = plt.subplots(figsize=(15, 7))
    n = len(points_list)
    hn = int(n/2)
    
    for i, landmark in enumerate(points_list):
        plt.subplot(2, hn, i+1)
        plt.imshow(img)
        plt.xticks(())
        plt.yticks(())   
        plt.plot(landmark[:,0], landmark[:,1], 'ro',  markersize=1)
     
    plt.show()
    
