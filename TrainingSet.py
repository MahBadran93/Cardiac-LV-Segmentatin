import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2 
import segment as seg


def shapeLandMark(trainingImage , sliceNum):
    # trainingImage : GT image file, contains 10 image slices 
    # sliceNum : number of slice 
    """
        this function return an annotated image called s, contains positions(x,y) as numpy array
    """
    slice1Copy = np.uint8(trainingImage[:,:,sliceNum])
    #ss = cv2.cvtColor(slice1Copy, cv2.COLOR_BGR2GRAY)
    edgedImg = cv2.Canny(slice1Copy,0,1)
    #print('testShapeLandMark')
    #plt.imshow(edgedImg)
    #plt.show()

    listOfIndex = np.where(edgedImg > 0)
    np.savetxt('tt.out', listOfIndex, delimiter='')
    arrayOfCoordinates = list(zip(listOfIndex[0], listOfIndex[1])) # combine two vectors(337) into one array of shape(337,2) 
    s = np.array(arrayOfCoordinates)
    return s
    



