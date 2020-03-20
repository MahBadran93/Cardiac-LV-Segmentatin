import numpy as np 
import cv2 
import matplotlib.pyplot as plt 

i = 0
# reteurn segmented GT image(segment only endocardium) 
def segmentGTEndy(gtImage,sliceNum):
    gtImage[:,:,sliceNum][gtImage[:,:,sliceNum] < 3] = 0
    slice1Copy = np.uint8(gtImage[:,:,sliceNum])
    newedgedImg = cv2.Canny(slice1Copy,0,1)
   # plt.imshow(newedgedImg)
    #plt.show()

    #cv2.imshow("segment",	newedgedImg)
    #cv2.waitKey(1000)
    listOfIndex = np.argwhere(newedgedImg != 0)
    #print(listOfIndex)
    

    #arrayOfCoordinates = list(zip(listOfIndex[0], listOfIndex[1])) # combine two vectors(337) into one array of shape(337,2) 
    #s = np.array(arrayOfCoordinates)
    return listOfIndex


    

