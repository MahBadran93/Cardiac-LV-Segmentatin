

import numpy as np
import cv2

from loader import DataLoader
from dentalvision.pdm.model import create_pdm
from dentalvision.glm.model import create_glm
from dentalvision.asm.model import ActiveShapeModel
from dentalvision.utils.multiresolution import gaussian_pyramid
from dentalvision.utils.structure import Shape
import loadnif as nif
import matplotlib.pyplot as plt
import translateinistialshape as tran
from dentalvision.utils.align import Aligner
import ROI as roi



LANDMARK_AMOUNT = 30            # nuber of landmarks per lv shape

MSE_THRESHOLD = 60000            # maximally tolerable error
    

def run(imageslice, centroid):
    '''
    Main method of the package.
    '''
        
    # ------------- LOAD DATA -------------- #
    loader = DataLoader()
    training_set, test_set = loader.leave_one_out(test_index=1)
    
    # --------------- TRAINING ---------------- #
    trainlandmarks = training_set[1]
 
    # build and train an Active Shape Model
    asm = ASMTraining(training_set, k=3, levels=3)
    pca=asm.activeshape.pdmodel
    
    t = 0
    for i in range(len(pca.eigenvalues)):
      if sum(pca.eigenvalues[:i]) / sum(pca.eigenvalues) < 0.99:
          t = t + 1
      else: break
      
    print ("Constructed model with {0} modes of variation".format(t))
    
    # --------------- TESTING ----------------- #
    aligner= Aligner()
    testimage, testlandmarks = test_set
    test= Shape(testlandmarks)
    # remove some noise from the test image
    testimage = remove_noise(testimage)
    plt.imshow(testimage) 
    plt.plot(test.x,test.y,'r.')
    plt.show()
    
       
    plt.imshow(imagestest)
    plt.plot(test.x,test.y,"r.")
    
    
    pose_para = aligner.get_pose_parameters(pca.mean,test)
    lst = list(pose_para)
    lst[0] = 0
    lst[1] = 0
    lst[2] = pose_para[2]
    lst[3] = 0
    t = tuple(lst)
    
    points = aligner.transform(pca.mean, t)
    
    meanShapeCentroid = (np.sum(points.x)/30,np.sum(points.y)/30)
    # centroid= tran.set_clicked_center(np.uint8(testimage))
    matches1 = tran.initalizeShape(centroid, meanShapeCentroid, points.matrix.T)
    if not isinstance(matches1, Shape):
        matches1= Shape(matches1.T)
        

    plt.imshow(imagestest)
    plt.plot(matches1.x,matches1.y,'.')
    plt.show()
    
    new_fit = asm.activeshape.multiresolution_search(imagestest, matches1, t=10, max_level=0, max_iter=20,n=0.1)
    # Find the target that the new fit represents in order
    # to compute the error. This is done by taking the smallest
    # MSE of all targets.
    mse = mean_squared_error(testlandmarks,new_fit)
        # implement maximally tolerable error
    if int(mse) < MSE_THRESHOLD: 
        print ('MSE:', mse)
           # plot.render_shape_to_image(np.uint8(testimage), trainlandmarks[best_fit_index], color=(0, 0, 0))
    else:
        print ('Bad fit. Needs to restart.')


class ASMTraining(object):
    '''
    Class that creates a complete Active Shape Model.
    The Active Shape Model is initialised by first building a point distribution
    model and then analysing the gray levels around each landmark point.
    '''
    def __init__(self, training_set, k=8, levels=4):
        self.images, self.landmarks = training_set#, self.landmarks_per_image
        # remove some noise from the image data
        for i in range(self.images.shape[0]):
            self.images[i] = remove_noise(self.images[i])

        print( '***Setting up Active Shape Model...')
        # 1. Train POINT DISTRIBUTION MODEL
        print ('---Training Point-Distribution Model...')
        pdmodel = self.pointdistributionmodel(self.landmarks)

        # 2. Train GRAYSCALE MODELs using multi-resolution images
        print ('---Training Gray-Level Model pyramid...')
        glmodel_pyramid = self.graylevelmodel_pyramid(k=k, levels=levels)

        # 3. Train ACTIVE SHAPE MODEL
        print ('---Initialising Active Shape Model...')
        self.activeshape = ActiveShapeModel(pdmodel, glmodel_pyramid)

        print ('Done.')

    def pointdistributionmodel(self, landmarks):
        '''
        Create model of shape from input landmarks
        '''
        return create_pdm(landmarks)

    def graylevelmodel(self, images, k=0, reduction_factor=1):
        '''
        Create a model of the local gray levels throughout the images.

        in: list of np array; images
            int k; amount of pixels examined on each side of the normal
            int reduction_factor; the change factor of the shape coordinates
        out: GrayLevelModel instance
        '''
        dd=np.asarray(self.landmarks)/reduction_factor
        
        print (dd.shape) 
        return create_glm(images, dd, k=k)

    def graylevelmodel_pyramid(self, levels=0, k=0):
        '''
        Create grayscale models for different levels of subsampled images.
        Each subsampling is done by removing half the pixels along
        the width and height of each image.

        in: int levels amount of levels in the pyramid
            int k amount of pixels examined on each side of the normal
        out: list of graylevel models
        '''
        # create Gaussian pyramids for each image
        multi_res = np.asarray([gaussian_pyramid(self.images[i], levels=levels) for i in range(self.images.shape[0])])
        # create list of gray-level models
        glmodels = []
        for l in range(levels):
            glmodels.append(self.graylevelmodel(multi_res[:, l], k=k, reduction_factor=2**l))
            print ('---Created gray-level model of level ' + str(l))
        return glmodels


def remove_noise(img):
    '''
    Blur image to partially remove noise. Uses a median filter.
    '''
    # cv2.threshold(img,150,255,cv2.THRESH_BINARY)
    return cv2.medianBlur(img, 5)



def mean_squared_error(landmark, fit):
    '''
    Compute the mean squared error of a fitted shape w.r.t. a
    test landmark.

    in: np array landmark
        Shape fit
    out: int mse
    '''
    
    aa=np.sum((fit.vector - landmark)**2)/fit.length
    
    return aa


