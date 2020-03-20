import numpy as np 
import cv2 
import matplotlib.pyplot as plt 


# reteurn segmented GT image(segment only endocardium) 
def segmentGTEndy(gtImage,sliceNum):
    gtImage[:,:,sliceNum][gtImage[:,:,sliceNum] < 3] = 0
    slice1Copy = np.uint8(gtImage[:,:,sliceNum])
    newedgedImg = cv2.Canny(slice1Copy,0,1)
        #cv2.imshow("segment",	newedgedImg)
        #cv2.waitKey(1000)
    return newedgedImg


    

