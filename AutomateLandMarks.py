
import loadnif as nif
import os
import numpy as np
import cv2
import PreProcess as preProc
import matplotlib.pyplot as plt
import SimpleITK as sitk


shapeList = []
c=0
# reteurn segmented GT image(segment only endocardium) 

def segmentGTEndy(gtImage,sliceNum):
    #im = sitk.GetArrayFromImage(gtImage[:,:,sliceNum]) 
    sampledImg = preProc.SampleTest1(gtImage[:,:,sliceNum])
    sampledImg[:,:][sampledImg[:,:] != 3] = 0
    slice1Copy = np.uint8(sampledImg[:,:])
    newedgedImg = cv2.Canny(slice1Copy,0,1)
    cv2.imshow('patient',newedgedImg)
    cv2.waitKey(1000)
    
    listOfIndex = np.argwhere(newedgedImg !=0)
    return listOfIndex

def getLandMarksCoords():
    path = '../training/'
    for root, dirs, files in os.walk(path): # 100 iteration, num of patients in training Folder
        dirs.sort()
        files.sort()
        for name in files: # iterate 6 times, depends on num of files 
            #sliceGT1 = nif.loadAllNifti(root,files[3:4].pop())
            #sliceGT2 = nif.loadAllNifti(root, files[5:6].pop())
            sliceGT1 = nif.loadNiftSimpleITK(root,files[3:4].pop())
            sliceGT2 = nif.loadNiftSimpleITK(root, files[5:6].pop())
            #print(simpitkImg.GetOrigin()) # to check the image size,spacing,origin
            # itereate depends on num of slices
            for i in range (sliceGT1.GetSize()[2]): #sliceGT1.shape[2]
                #shapeList.append(segmentGTEndy(sliceGT1,i))
                shapeList.append(segmentGTEndy(sliceGT1,i))
                # array of positions(landmarks)(337 row ,2 column), vector in our shape
                #listCoords.append(seg.segmentGTEndy(sliceGT1,i).shape)
            for i in range (sliceGT2.GetSize()[2]): #
                #shapeList.append(segmentGTEndy(sliceGT2,i))
                shapeList.append(segmentGTEndy(sliceGT2,i))
                #listCoords.append(seg.segmentGTEndy(sliceGT2,i).shape)
                #print(listCoords)
            break
    
    return shapeList    

