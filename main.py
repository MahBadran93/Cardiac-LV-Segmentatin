

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
import Image_preperation as prep
import fitModel as fi
from dentalvision.utils.align import Aligner
import ROI as roi







LANDMARK_AMOUNT =30           # amount of landmarks per tooth

MSE_THRESHOLD = 60000            # maximally tolerable error


def remove_noise(img):
    '''
    Blur image to partially remove noise. Uses a median filter.
    '''
    cv2.threshold(img,150,255,cv2.THRESH_BINARY)
    return cv2.medianBlur(img, 5)


# def mean_squared_error(landmark, fit):
#     '''
#     Compute the mean squared error of a fitted shape w.r.t. a
#     test landmark.

#     in: np array landmark
#         Shape fit
#     out: int mse
#     '''
#     return np.sum((fit.vector - landmark)**2)/fit.length





class ASMTraining(object):
    '''
    Class that creates a complete Active Shape Model.
    The Active Shape Model is initialised by first building a point distribution
    model and then analysing the gray levels around each landmark point.
    '''
    
    def __init__(self, training_set, k=8, levels=4):
        self.images, self.landmarks = training_set #, self.landmarks_per_image 
        # remove some noise from the image data
        for i in range(self.images.shape[0]):
            self.images[i] = remove_noise(self.images[i])

        print ('***Setting up Active Shape Model...')
        # 1. Train POINT DISTRIBUTION MODEL
        print ('---Training Point-Distribution Model...')
        self.pdmodel = self.pointdistributionmodel(self.landmarks)

        #2. Train GRAYSCALE MODELs using multi-resolution images
        print ('---Training Gray-Level Model pyramid...')
        glmodel_pyramid = self.graylevelmodel_pyramid(k=k, levels=levels)

        # 3. Train ACTIVE SHAPE MODEL
        print ('---Initialising Active Shape Model...')
        self.activeshape = ActiveShapeModel(self.pdmodel,glmodel_pyramid)

        print ('Done.')
        
    def get_model(self):
        return self.pdmodel

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
    
        return create_glm(images, np.asarray(self.landmarks)/reduction_factor, k=k)

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
    

aligner= Aligner()


def run(img_edge,centroid,img):

    
    # ------------- LOAD DATA -------------- #
    loader = DataLoader()
    training_set, test_set = loader.leave_one_out(test_index=0)
    
    # --------------- TRAINING ---------------- #
    trainlandmarks = training_set[1]

    # build and train an Active Shape Model
    asm = ASMTraining(training_set, k=3, levels=3)
    pca=asm.activeshape.pdmodel
    
    t = 0
    for i in range(len(pca.eigenvalues)):
      if sum(pca.eigenvalues[:i]) / sum(pca.eigenvalues) < 0.98:
          t = t + 1
      else: break
      
    print ("Constructed model with {0} modes of variation".format(t))

    # --------------- TESTING ----------------- #
    
    
    testimage, testlandmarks = test_set
    test= Shape(testlandmarks)
  
    pose_para = aligner.get_pose_parameters(pca.mean,test)
    lst = list(pose_para)
    lst[0] = 0
    lst[1] = 0
    lst[2] = pose_para[2]
    lst[3] = 0
    
    t = tuple(lst)
    
    points = aligner.transform(pca.mean, t)
    
    plt.imshow(img)
    plt.plot(points.x,points.y,"r.")
    

    meanShapeCentroid = (np.sum(points.x)/30,np.sum(points.y)/30)
    #perform manual centroid 
    # centroid= tran.set_clicked_center(np.uint8(testimage))
    matches1 = tran.initalizeShape(centroid, meanShapeCentroid, points.matrix.T)
    if not isinstance(matches1, Shape):
        matches1= Shape(matches1.T)
        
    
    # meanShapeCentroid = (np.sum(testlandmarks.x)/30,np.sum(testlandmarks.y)/30)
    plt.imshow(img)
    plt.plot(matches1.x,matches1.y,'.')
    plt.show()
    
    
    x, y, new_p= active_shape(img_edge,matches1,pca,img)

    
    
    
    
    # def show_evolution(img, points_list):
        
    #     plt.figure()
    #     fig, ax = plt.subplots(figsize=(15, 7))
    #     n = len(points_list)
    #     hn = int(n/2)
        
    #     for i, landmark in enumerate(points_list):
    #         plt.subplot(2, hn, i+1)
    #         plt.imshow(img)
    #         plt.xticks(())
    #         plt.yticks(())   
    #         plt.plot(landmark[:,0], landmark[:,1], 'ro',  markersize=1)
         
    #     plt.show()
    
def active_shape(img_edge, init_shape, pca,img, length=10):
    'edge _img ,  pca_tooth : from prepratopn function '
    'tooth_point  ininitial position'

    new_points,error = fi.fit_measure(init_shape.matrix.T,length,img_edge)
    new_point11=Shape(new_points)
    

    b, pose_param =fi.match_model_points(new_points, pca)
    
    x = fi.generate_model_point(b, pca)
    x= Shape(x)
    x = aligner.transform(x, pose_param)

    
    meanShapeCentroid = (np.sum(x.x)/30,np.sum(x.y)/30)
  # centroid= tran.set_clicked_center(np.uint8(testimage))
    res = tran.initalizeShape(centroid, meanShapeCentroid, x.matrix.T)
    res= Shape(res.T)
    
        
    plt.imshow(img)
    plt.plot(res.x,res.y,'r.')
    plt.show()

   
    y = aligner.invert_transform(x,pose_param)
    y = aligner.transform(y, pose_param)
    


  #   meanShapeCentroid = (np.sum(y.x)/30,np.sum(y.y)/30)
  # # centroid= tran.set_clicked_center(np.uint8(testimage))
  #   res1 = tran.initalizeShape(centroid, meanShapeCentroid, y.matrix.T)
  #   y= Shape(res1.T)
    
  #   plt.imshow(imageslice)
  #   plt.plot(y.x,y.y,'r.')
  #   plt.show()
    


    return x, y , new_point11


if __name__ == '__main__':
    
      
    path = '../training/patient050/patient050_frame01.nii.gz'
    simpitkImgSys = nif.loadSimpleITK(path)
    
    imageslice= nif.getNotNumpySliceITK(simpitkImgSys,4)
    imagestest = nif.getSliceITK(simpitkImgSys,4)
    spacing =imageslice.GetSpacing()[0]
    loadNiftinibabel=nif.loadNifti(path)
    centroid = roi.extract_roi(loadNiftinibabel, spacing)
    imagestest = remove_noise(imagestest)
    idge_canny =prep.calc_external_img_active_contour(imagestest)
    edge_sobel=prep.sobel(imagestest)
    
    run(edge_sobel,centroid,imagestest)






