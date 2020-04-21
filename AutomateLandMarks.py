
#....................Important Packages........................................
import loadnif as nif
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk
from shapely.geometry import Polygon 
import  scipy.spatial as align
#..............................................................................

class LandMarks():

    '''
        all operation of extracting shape landmarks from the ground truth are done in this class
    '''
      
    def __init__(self):
        
        self.shapeList = [] # list to include all the shapes coords
        self.shapeCentroids = []  
        self.count = 0            
        self.fixedCentroidS = []
        self.listOfIndex = []
        self.translatedShape = []
        self.LandmarkedShapes = [] # store landmark for all slice 
        self.fixedShape = []       
        self.alignedList = []

        
    def single_parametric_interpolate(self,obj_x_loc,obj_y_loc,numPts=30):
        '''
        Parameters
        ----------
        obj_x_loc : X-coords of the shape we want to sample
        obj_y_loc : Y-coords of the shape we want to sample
        numPts : Number of sampled points in each shape
            DESCRIPTION. The default is 30.

        Returns
        -------
        new_points : Returns the new 30 sampled points for shape.
        This function resample each shape to same number of landmarks 

        '''
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
        Parameters
        ----------
        gtImage : the Ground Truth image in the DataSet which is a 3D shape,(W,H,SliceNim)
        sliceNum : Number of the slic 

        Returns
        -------
        This function applis canny edge detection algorithm to detect the contour of LV endocardium,
        and extracts the coordinates for the LV contour, so it returns a list of coordinates
        for the endocardium detected contour

        '''
        sampledImgArr = sitk.GetArrayFromImage(gtImage[:,:,sliceNum])
        sampledImgArr[:,:][sampledImgArr[:,:] != 3] = 0
        slice1Copy = np.uint8(sampledImgArr[:,:])
        newedgedImg = cv2.Canny(slice1Copy,0,1)
        self.listOfIndex = np.argwhere(newedgedImg !=0)

        '''
        To Display and show the shapes 
        cv2.imshow('patient',newedgedImg)
        cv2.waitKey(1000)
        '''
        
        return self.listOfIndex # return all coordinates of the shape 
    
    
    def getShapeCoords(self):
        '''
        Returns
        -------
        This function loops over all the ACDC MRI dataSet and returns a list 
        that contatins of all the slice shapes with 1788 length(1788 shappes), but 
        the shapes in this list are not aligned or sampled
            
        '''
        path = '../training/'
        for root, dirs, files in os.walk(path): # 100 iteration, num of patients in training Folder
            dirs.sort()
            files.sort()
            for name in files: # iterate 6 times, depends on num of files 
                #sliceGT1 = nif.loadAllNifti(root,files[3:4].pop())
                #sliceGT2 = nif.loadAllNifti(root, files[5:6].pop())
                sliceGT1 = nif.loadNiftSimpleITK(root,files[3:4].pop())
                sliceGT2 = nif.loadNiftSimpleITK(root, files[5:6].pop())
                # itereate depends on num of slices
                for i in range (sliceGT1.GetSize()[2]): #sliceGT1.shape[2]
                    self.shapeList.append(self.extractContourCoords(sliceGT1,i) )
                    #self.shapeCentroids.append(self.findCentroid(self.extractContourCoords(sliceGT1,i))) 
                for i in range (sliceGT2.GetSize()[2]): #
                    #shapeList.append(extractContourCoords(sliceGT2,i))
                    self.shapeList.append(self.extractContourCoords(sliceGT2,i))
                    #self.shapeCentroids.append(self.findCentroid(self.extractContourCoords(sliceGT2,i)))  
                break
        return self.shapeList 
    
    def GenerateSampleShapeList(self):
        '''
        This function returns a list that contains a sampled and aligned data,
        list length is 1788 shape.

        Returns
        -------
        alignedList :
            list of sampled and aligned shapes
        originalShapes : 
            list of unsampled and unaligned shapes .

        '''
        
        originalShapes = self.getShapeCoords() #return usampled list of shapes
        for i in range(len(originalShapes)):
            if(len(originalShapes[i]) !=0):
                
                "interpolate the point contour to fit a polygon"
                poly = Polygon([p[0],p[1]] for p in originalShapes[i])
                x,y = poly.convex_hull.exterior.coords.xy
                
                "sample each shape ((polygon) with 30 point lanmark"
                SampledShape = np.array(self.single_parametric_interpolate(x,y,numPts=30))
                
                # x_sampled = [p[0] for p in SampledShape]
                # y_sampled = [p[1] for p in SampledShape]
                
                self.LandmarkedShapes.append(SampledShape)
                
                if(i==1): # we created a flag i to fix a reference shape to align all the shapes to 
                    self.fixedShape = SampledShape # reference shape
                
                mtx1, mtx2, disparity = align.procrustes(self.fixedShape,SampledShape) # return Transformation parameters and aligned shape
                self.alignedList.append(mtx2)
                #print(30-len(x))
           
            else:    
                print('empty shape')
            
            
                
        return self.alignedList, originalShapes 
    
    
    
    
    
    
    
    
    
    
    # our own implementation to align data but we choose already implenmented function from scipy 
    '''
    if(self.count == 1): # create fixed contour
        self.fixedCentroidS = self.findCentroid(self.listOfIndex) # Create fixed Centroid for all iages 
        self.count += 1
        
    if(len(self.listOfIndex) == 2): # to check shapes with points less than 2.
        continue
        #self.translatedShape = self.translateShapesToFixedCentroid(self.fixedCentroidS,self.listOfIndex)
        
        
    
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
    '''  
        
        
    
    