import os
import nibabel as nib
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2 
import TrainingSet as train
from nibabel.testing import data_path 
import segment as seg


#########################################################################

def loadNifti(path): # this class returns nifti image array numpy type
    img1 = nib.load(path)
    trainingImg = img1.get_fdata() # 4d, to show the animated cardiac image
    return trainingImg 


def loadAllNifti(root , file): # this class returns nifti image array numpy type
    path =  root + '/' + file 
    img1 = nib.load(path)
    trainingImg = img1.get_fdata() # 4d, to show the animated cardiac image
    return trainingImg     
###########################################################################

def displayAnimatedNifti(niftiImage, sliceNum1):
    slicePat1 = [] # initalizing a list 
    for t in range(niftiImage.shape[3]): # 30 frames, the image shape (width, heigh, # of slices , 30 frame for each slice), thats why we put 30 parameter for our array that holds 30 frame for slice 0 to animate
        slicePat1.append(niftiImage[:, :, sliceNum1,t])    
    fig = plt.figure() # make figure
    axes = fig.subplots(1,2) # make two subplots in the figure
    im = axes[1].imshow(slicePat1[0], vmin=0, vmax=255,cmap="gray", origin="lower") #show 3d animated image
    def updatefig(j):
        im.set_array(slicePat1[j])
        return [im]
    #axes[0].imshow(niftiImage[:,:,9])   #show segmented slice
    ani = animation.FuncAnimation(fig, updatefig, frames=range(np.array(slicePat1).shape[0]), 
                              interval=50, blit=True)
    plt.show()   

"""
niftiImage : image loaded from nii.gz file 4d, it has 10 slices, from 0-9 , each slice with 30 frame animation.
sliceNum1 : number of slice from 0 - 9
"""
##############################################################################
def displaySlices(imageGT,sliceNum):
    plt.imshow(getSlice(imageGT,sliceNum))
    plt.show()
"""
imageGT : image loaded fron GT file 
sliceNum : it has 10 slices , from 0-9 
"""
###############################################################################

def getSlice(image,numOfSlice):
    if(numOfSlice >= image.shape[2]):
        print("number of slices is only", image.shape[2])
        return 0
    else:    
        return image[:,:,numOfSlice]
"""
return segmented slices individually for every specific image  
"""
#################################################################################

img4D = loadNifti('../training/patient001/patient001_4d.nii.gz')
imgES= loadNifti('../training/patient001/patient001_frame12.nii.gz')
imgED =loadNifti('../training/patient001/patient001_frame01.nii.gz') 
imgESGT = loadNifti('../training/patient002/patient002_frame12_gt.nii.gz')
strokeVolume = imgED[:,:,5] - imgES[:,:,5]
####################################################################333
#........... Create Training Set...........................
shapeList = []
listCoords = []
listOfDim = []
def LoadAllGT():

    path = '../training/'
    for root, dirs, files in os.walk(path): # 100 iteration, num of patients in training Folder
        dirs.sort()
        files.sort()
        for name in files: # iterate 6 times, depends on num of files 
            sliceGT1 = loadAllNifti(root,files[3:4].pop())
            sliceGT2 = loadAllNifti(root, files[5:6].pop())


            # itereate depends on num of slices
            for i in range (sliceGT1.shape[2]):
                shapeList.append(seg.segmentGTEndy(sliceGT1,i)) 
                listOfDim.append(len(shapeList[i]))
                # array of positions(landmarks)(337 row ,2 column), vector in our shape
                #listCoords.append(seg.segmentGTEndy(sliceGT1,i).shape)
            for i in range (sliceGT2.shape[2]):
                shapeList.append(seg.segmentGTEndy(sliceGT2,i))
                #print(shapeList[i].shape)
                listOfDim.append(len(shapeList[i]))

                #listCoords.append(seg.segmentGTEndy(sliceGT2,i).shape)
                #print(listCoords)
            break
        
#return finalShapeArr
LoadAllGT()
#print(np.array(shapeList[1800])[:,1])
                
plt.plot(np.array(shapeList[4])[:,0], np.array(shapeList[4])[:,1], '.')
plt.axis([-500, 500, -500, 500])
plt.show()


#print(LoadAllGT().shape)        
            

#################################################################

#for i in range(imgGT.shape[2]):
    #displaySlices(imgGT,i) 
#displaySegmentedGTSlices(imgRe,5)
