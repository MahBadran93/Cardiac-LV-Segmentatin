import os
import nibabel as nib
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation



from nibabel.testing import data_path 


#########################################################################

def loadNifti(path): # this class returns nifti image array numpy type
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

imgRe = loadNifti('../training/patient001/patient001_frame12.nii.gz')
imgGT = loadNifti('../training/patient001/patient001_frame12_gt.nii.gz')

print(imgRe.shape , imgGT.shape)
#displayAnimatedNifti(imgRe,5)

#getSlice(imgRe,9)
for i in range(imgGT.shape[2]):
    displaySlices(imgGT,i) 
#displaySegmentedGTSlices(imgRe,5)

#print(imgGT.shape)








"""
def show_slices(slices):
    fig, axes = plt.subplots(1,len(slices))
    
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        
"""        

"""
plt.imshow(trainingImg2[:,:,6]) # show the segmented slices 
plt.show()
"""
#array = np.asarray(trainingImg1)
#arraych = np.reshape(array, [216,256,1])
"""
slice_0 = trainingImg1[:, :, 0,29]
slice_1 = trainingImg1[:, :, 1,29]
slice_2 = trainingImg1[:, :, 2,29]
slice_3 = trainingImg1[:, :, 3,29]
slice_4 = trainingImg1[:, :, 4,29]
slice_5 = trainingImg1[:, :, 5,29]
slice_6 = trainingImg1[:, :, 6,29]
slice_7 = trainingImg1[:, :, 7,29]
slice_8 = trainingImg1[:, :, 8,29]
slice_9 = trainingImg1[:, :, 9,29]
"""

#print(slicePat1[0])




#cmap=plt.get_cmap('jet')




#ani = animation.FuncAnimation(fig, updatefig, frames=range(20), 
                          #    interval=50, blit=True)



"""
for y in range(29):
    plt.imshow(slicePat1[y])
    plt.show()
"""
