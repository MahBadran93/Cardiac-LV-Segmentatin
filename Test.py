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
import  SimpleITK as sitk

from sklearn.decomposition import PCA

#.............................Testing.....................................................

#........... Create Training Model...........................
showLand = LandMarks()
landMarkedShapes,originalShape, sampledwithoutAlighn= showLand.GenerateSampleShapeList()

#plot original unsampled list of shapes 
s = 0.5
shift = 50

for i in range (len(landMarkedShapes)):
    x1 = [p[0] for p in landMarkedShapes[i]]
    y1 = [p[1] for p in landMarkedShapes[i]]
    plt.plot(x1,y1,'.')
    plt.show()
path = '../training/patient001/patient001_frame01_gt.nii.gz'

x1 = [p[0] for p in originalShape[7]]
y1 = [p[1] for p in originalShape[7]]
x2 = [p[0] for p in sampledwithoutAlighn[6]]
y2 = [p[1] for p in sampledwithoutAlighn[6]]
x3 = [p[0] for p in landMarkedShapes[6]]
y3 = [p[1] for p in landMarkedShapes[6]]


image = nif.loadSimpleITK(path)
image2 = nif.loadNifti(path)

imageslice = nif.getSliceITK(image,8)
imageslice2 = nif.getSlice(image2,8) 
plt.imshow(imageslice)
plt.plot(x1,y1,'.')
plt.plot(x2,y2)

plt.plot(x3,y3)



plt.show()



'''
# Plot the list of all the shapes 
for i in range (len(landMarkedShapes)):
    plt.axis([-0.4, 0.6, -0.4, 0.6])
    x1 = [p[0] for p in landMarkedShapes[i]]
    y1 = [p[1] for p in landMarkedShapes[i]]
    plt.plot(x1,y1)
'''
# convert the list to numpy array 
landMarkedShapesR = np.stack(landMarkedShapes,axis=0)
landMarkedShapesR2 = np.array(landMarkedShapes).T

# final Matrix of Sampled & aligned shapes, Diminsion (2,30,1788) 
LandMarkFinalMatrix =landMarkedShapesR.T

# Reshape the final matrix to Diminsion (2,1788, 30)
AlignedMAtrixOfShapes = np.transpose(LandMarkFinalMatrix,(0,2,1)) 
'''
# plot the sampled and aligned list of shapes (Numpy array)
for i in range(AlignedMAtrixOfShapes.shape[1]):
    x1 = AlignedMAtrixOfShapes[0,i,:]
    y1 = AlignedMAtrixOfShapes[1,i,:]
    plt.axis([-0.4, 0.6, -0.4, 0.6])
    plt.plot(x1,y1)
    plt.show()
'''    

    
    


