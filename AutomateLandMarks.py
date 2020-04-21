
import loadnif as nif
import os
import numpy as np
import cv2
import PreProcess as preProc
import matplotlib.pyplot as plt
import SimpleITK as sitk
from shapely.geometry import Polygon 

<<<<<<< HEAD
import AlignShape as al

#from scipy.spatial import procrustes

=======
>>>>>>> master




class LandMarks():
    # all operation of extracting land mark from the ground truth done in this class
    
    
    def __init__(self):
        
        self.shapeList = [] # list to include all the shapes coords
        self.shapeCentroids = []  
        self.count = 0            
        self.fixedCentroidS = []
        self.listOfIndex = []
        self.translatedShape = []
        self.LandmarkedShapes = [] # store landmark for all slice 
        
        # reteurn segmented GT image(segment only endocardium) 
    
    # this function resample each shape to same number of landmarks 
    def single_parametric_interpolate(self,obj_x_loc,obj_y_loc,numPts=30):
        n = len(obj_x_loc)
        vi = [[obj_x_loc[(i+1)%n] - obj_x_loc[i],
             obj_y_loc[(i+1)%n] - obj_y_loc[i]] for i in range(n)]
        si = [np.linalg.norm(v) for v in vi]
        di = np.linspace(0, sum(si), numPts, endpoint=False)
        new_points = []
        for d in di:
            for i,s in enumerate(si):
                if d>s: d -= s
                else: break
            l = d/s
            new_points.append([obj_x_loc[i] + l*vi[i][0],
                               obj_y_loc[i] + l*vi[i][1]])
        return new_points
    
    
    def extractContourCoords(self,gtImage,sliceNum):
        
        '''
        gtImage: ground truth image  from End_systolic and End_diastolic 
        sliceNum: slice number from each volume
        
        return position coordinates for each contour

        '''
        #im = sitk.GetArrayFromImage(gtImage[:,:,sliceNum]) 
        #sampledImg = preProc.SampleTest1(gtImage[:,:,sliceNum]) # register the image 
        sampledImgArr = sitk.GetArrayFromImage(gtImage[:,:,sliceNum])
        sampledImgArr[:,:][sampledImgArr[:,:] != 3] = 0
        slice1Copy = np.uint8(sampledImgArr[:,:])
        newedgedImg = cv2.Canny(slice1Copy,0,1)
        #cv2.imshow('patient',newedgedImg)
        #cv2.waitKey(1000)
        self.listOfIndex = np.argwhere(newedgedImg !=0)
        
<<<<<<< HEAD
        
        
        '''
=======
>>>>>>> master
        if(self.count == 1): # create fixed contour
            self.fixedCentroidS = self.findCentroid(self.listOfIndex) # Create fixed Centroid for all iages 
        self.count += 1
        
        if(len(self.listOfIndex) > 2): # to check shapes with points less than 2.
            self.translatedShape = self.translateShapesToFixedCentroid(self.fixedCentroidS,self.listOfIndex)
<<<<<<< HEAD
        '''
        return self.listOfIndex # return all coordinates of the shape 
=======
            
        return self.translatedShape # return all coordinates of the shape 
>>>>>>> master
    
    def findCentroid(self, shapeI):
        imgShape = np.array(shapeI)
        length = imgShape.shape[0] # the same length (128,128) size of the image
        sum_x = np.sum(imgShape[:, 0])
        sum_y = np.sum(imgShape[:, 1])
        return [np.round(sum_x/length), np.round(sum_y/length)] # returm coords of the centroid for each shape 
    
    def translateShapesToFixedCentroid(self,FixedshapeCentroid, shape):
        # FixedshapeCentroid : Fixed Centroid Value , i.e : (120,120).
        # shape : the shape to center around FixedshapeCentroid. 
        subtractValue = np.subtract(np.array(FixedshapeCentroid),np.array(self.findCentroid(shape)))
        translatedShpe = np.add(shape,subtractValue)
        
        return translatedShpe # return a translated shape around a specific fixed centroid for all shapes 
      
<<<<<<< HEAD
  
=======
        
        
    
>>>>>>> master
    def getShapeCoords(self):
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
                    #shapeList.append(extractContourCoords(sliceGT1,i))

                    self.shapeList.append(self.extractContourCoords(sliceGT1,i))
                    #self.shapeCentroids.append(self.findCentroid(self.extractContourCoords(sliceGT1,i))) 
                    # array of positions(landmarks)(337 row ,2 column), vector in our shape
                    #listCoords.append(seg.extractContourCoords(sliceGT1,i).shape)
                for i in range (sliceGT2.GetSize()[2]): #
                    #shapeList.append(extractContourCoords(sliceGT2,i))
                    self.shapeList.append(self.extractContourCoords(sliceGT2,i))
                    #self.shapeCentroids.append(self.findCentroid(self.extractContourCoords(sliceGT2,i)))  
                    #listCoords.append(seg.extractContourCoords(sliceGT2,i).shape)
                    #print(listCoords)
                break
        return self.shapeList 
    
    def GenerateSampleShapeList(self):
        
        listInit = self.getShapeCoords() #return usampled list of shapes        
        for i in range(len(listInit)):
            if(len(listInit[i]) !=0):
          #       plt.figure()
          #     x1 = [p[0] for p in listInit[i]]
          #     y1 = [p[1] for p in listInit[i]]
                
                "interpolate the point contour to fit a polygon"
                poly = Polygon([p[0],p[1]] for p in listInit[i])
                x,y = poly.convex_hull.exterior.coords.xy
                
                "sample each shape ((polygon) with 30 point lanmark"
                SampledShape = np.array(self.single_parametric_interpolate(x,y,numPts=30))
                
<<<<<<< HEAD
                if i==1:
                    fixshape = SampledShape

         #       x_sampled = [p[0] for p in SampledShape]
         #       y_sampled = [p[1] for p in SampledShape]
                
                
                if i>1:
                    d,z,t=al.procrustes(fixshape,SampledShape)
                #print(30-len(x))
                    
                self.LandmarkedShapes.append(SampledShape)

=======
         #       x_sampled = [p[0] for p in SampledShape]
         #       y_sampled = [p[1] for p in SampledShape]
                
                self.LandmarkedShapes.append(SampledShape)
                
                #print(30-len(x))
>>>>>>> master
                
                #plt.axis([-216, 304, -216, 304])
                #plt.plot(x_sampled,y_sampled)
                
                #plt.plot(x,y)
                #print(findCentroid(x,y))
                #print(len(x))
                #t,y = poly.convex_hull.coords.xy
                #t,n = poly.contour.exterior.coords.xy
                
                #plt.show()
                #print(len(SampledShape))
            else:    
                print('empty shape')
                
<<<<<<< HEAD
                
        
                
        return self.LandmarkedShapes, listInit, d,z,t
    
    


    

    
    


=======
        return self.LandmarkedShapes, listInit   
>>>>>>> master



