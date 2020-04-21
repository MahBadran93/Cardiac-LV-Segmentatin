
import loadnif as nif
import os
import numpy as np
import cv2
import PreProcess as preProc
import matplotlib.pyplot as plt
import SimpleITK as sitk
from PIL import Image


shapeList = []
# reteurn segmented GT image(segment only endocardium) 
i = 0
def SaveToFolder(gtImage,sliceNum, imgNumFolder):
    sampledImg = preProc.SampleTest1(gtImage[:,:,sliceNum])
    #print(sampledImg.GetSpacing())
    sampledImgArr = sitk.GetArrayFromImage(sampledImg)
    #slice1Copy = np.uint8(sampledImgArr)
    path = './TrainingImages'
    cv2.imwrite(os.path.join(path , 'testImage{0}.png'.format(imgNumFolder)), sampledImgArr)
    cv2.waitKey(0)

    #cv2.imshow('mmm', slice1Copy)
    #cv2.waitKey(500)
    #plt.imshow(sampledImgArr)
    #plt.show()

def saveDataSet():
    path = '../training/'
    count = 0
    for root, dirs, files in os.walk(path): # 100 iteration, num of patients in training Folder
        dirs.sort()
        files.sort()
        for name in files: # iterate 6 times, depends on num of files 
            simpitkImg = nif.loadNiftSimpleITK(root,files[2:3].pop())
            simpitkImg2 = nif.loadNiftSimpleITK(root,files[4:5].pop())
            # itereate depends on num of slices
            for i in range (simpitkImg.GetSize()[2]): #sliceGT1.shape[2]
                count = count + 1
                SaveToFolder(simpitkImg, i, count)
                break
            for i in range (simpitkImg2.GetSize()[2]): 
                count = count + 1
                SaveToFolder(simpitkImg2,i,count)
                break
            break
    

