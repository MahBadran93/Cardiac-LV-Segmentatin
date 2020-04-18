
import loadnif as nif
import os
import numpy as np
import cv2
import PreProcess as preProc
import matplotlib.pyplot as plt
import SimpleITK as sitk




class LandMarks():
    
    
    def __init__(self):
        self.shapeList = [] # list to include all the shapes coords
        self.shapeCentroids = []
        c=0
        # reteurn segmented GT image(segment only endocardium) 
    
    
    def segmentGTEndy(self,gtImage,sliceNum):
        #im = sitk.GetArrayFromImage(gtImage[:,:,sliceNum]) 
        #sampledImg = preProc.SampleTest1(gtImage[:,:,sliceNum]) # register the image 
        sampledImgArr = sitk.GetArrayFromImage(gtImage[:,:,sliceNum])
        sampledImgArr[:,:][sampledImgArr[:,:] != 3] = 0
        slice1Copy = np.uint8(sampledImgArr[:,:])
        newedgedImg = cv2.Canny(slice1Copy,0,1)
        #cv2.imshow('patient',newedgedImg)
        #cv2.waitKey(1000)
    
        listOfIndex = np.argwhere(newedgedImg !=0)
        return listOfIndex # return all coordinates of the shape 
    
    def findCentroid(self, img):
        length = img.shape[0] # the same length 128,128 size of the image
        sum_x = np.sum(img[:, 0])
        sum_y = np.sum(img[:, 1])
        return [np.round(sum_x/length), np.round(sum_y/length)] # returm coords of the centroid for each shape 

  
        
    
    def getLandMarksCoords(self):
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
                    self.shapeList.append(self.segmentGTEndy(sliceGT1,i))
                    self.shapeCentroids.append(self.findCentroid(self.segmentGTEndy(sliceGT1,i))) 
                    # array of positions(landmarks)(337 row ,2 column), vector in our shape
                    #listCoords.append(seg.segmentGTEndy(sliceGT1,i).shape)
                for i in range (sliceGT2.GetSize()[2]): #
                    #shapeList.append(segmentGTEndy(sliceGT2,i))
                    self.shapeList.append(self.segmentGTEndy(sliceGT2,i))
                    self.shapeCentroids.append(self.findCentroid(self.segmentGTEndy(sliceGT2,i)))  
                    #listCoords.append(seg.segmentGTEndy(sliceGT2,i).shape)
                    #print(listCoords)
                break
        
        return self.shapeList, self.shapeCentroids   
    



