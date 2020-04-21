from AutomateLandMarks import  LandMarks
import saveDataSet as saveData
import loadnif as nif
import numpy as np
import matplotlib.pyplot as plt
import cv2
import  shapely as shp
import shapely.geometry as gmt
import pylab as pl
import  scipy as sc

from sklearn.decomposition import PCA

#.............................Testing.....................................................

#........... Create Training Model...........................
showLand = LandMarks()
landMarkedShapes,originalShape= showLand.GenerateSampleShapeList()

# Plot the list of all the shapes 
for i in range (len(landMarkedShapes)):
    plt.axis([-0.4, 0.6, -0.4, 0.6])
    x1 = [p[0] for p in landMarkedShapes[i]]
    y1 = [p[1] for p in landMarkedShapes[i]]
    plt.plot(x1,y1)

# convert the list to numpy array 
landMarkedShapesR = np.stack(landMarkedShapes,axis=0)
landMarkedShapesR2 = np.array(landMarkedShapes).T
LandMarkFinalMatrix =landMarkedShapesR.T

# final Matrix of Sampled & aligned shapes, Diminsion (2,1788,30) 
AlignedMAtrixOfShapes = np.transpose(LandMarkFinalMatrix,(0,2,1)) 

# plot the converted list(Numpy array)
for i in range(AlignedMAtrixOfShapes.shape[1]):
    x1 = AlignedMAtrixOfShapes[0,i,:]
    y1 = AlignedMAtrixOfShapes[1,i,:]
    plt.axis([-0.4, 0.6, -0.4, 0.6])
    plt.plot(x1,y1)


