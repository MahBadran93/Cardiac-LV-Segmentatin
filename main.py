'''
Main file to run the computer vision package that detects
incisors in radiographs based on feature detection and
active shape models.

Run the program by calling:
    $ python main.py
If encountering problems, check whether the variables
that point to the image and landmark directories are correct.
See the file loader.py in this respect.

This file maintains the following procedure:
1. Loading:
    Loads data from the input constants in loader.py.
    Returns training and test sets from the image files and
    the landmark files.
    The image data sets is then blurred with a median filter
    to remove some noise of the radiographs.

2. Model SETUP:
    Two systems are trained. For initialisation, a feature
    detector is used that can switch between semi-automatic
    to fully automated search. Semi-automated search involved
    a search in a restricted space. Fully automated search
    uses a trained approximation of  restricted search space.
    Then, an Active Shape Model is trained using the loaded
    radiograph data. The model creates a shape model and a model
    of the gray level profiles around each model point.

3. Test environment:
    The models are tested by first scanning the image for
    matching regions using the feature detector and then by
    initialising the active shape model on the detected region.

@authors: Tina Smets, Tom De Keyser
'''
import numpy as np
import cv2

from loader import DataLoader
from dentalvision.pdm.model import create_pdm
from dentalvision.glm.model import create_glm
from dentalvision.asm.model import ActiveShapeModel
from dentalvision.featuredetect.model import create_featuredetector
from dentalvision.utils.multiresolution import gaussian_pyramid
from dentalvision.utils.structure import Shape
import loadnif as nif
import matplotlib.pyplot as plt
import translateinistialshape as tran

import Image_preperation as prep
from dentalvision.utils import plot



import fit
from model import Build_Model

from dentalvision.asm.fit import Fitter
from dentalvision.utils.align import Aligner












MATCH_DIM = (320, 110)          # dimensions searched by feature detector
LANDMARK_AMOUNT =30           # amount of landmarks per tooth

MSE_THRESHOLD = 60000            # maximally tolerable error
path = '../training/patient001/patient001_frame01.nii.gz'


def remove_noise(img):
    '''
    Blur image to partially remove noise. Uses a median filter.
    '''
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


class FDTraining(object):
    '''
    Class that trains a feature detecting system based on an eigen
    model of incisors and on the computation of a suitable search region.
    '''
    def __init__(self):
        print ('***Setting up Feature Detector...')
        print ('---Training...')
        #self.detector = create_featuredetector()

    def scan_region(self, landmarks, diff=0, searchStep=30):
        '''
        Scan landmark centroids to find a good search region for the feature
        detector.

        in: np array of landmarks Shapes
            int diff; narrows down the search space
            int seachStep; a larger search step fastens search, but also
                increases the risk of missing matches.
        '''
        centroids = np.zeros((landmarks.shape[0], 2))
        for l in range(landmarks.shape[0]):
            centroids[l] = Shape(landmarks[l]).centroid()

        x = (int(min(centroids[:, 0])) + diff, int(max(centroids[:, 0])) - diff)
        y = (int(min(centroids[:, 1])) + diff, int(max(centroids[:, 1])) - diff)

        return (y, x, searchStep)

    def match(self, image, match_frame=MATCH_DIM):
        '''
        Perform feature matching on image in the defined search region.
        Uses the specified target dimension as match region.

        Returns LANDMARK_AMOUNT points along the ellipse of each match. These
        points facilitate alignment with the ASM mean model.

        in: np array image
            tup(tup(int x_min, int x_max), tup(int y_min, int y_max), int searchStep)
                search region; defines the boundaries of the search
            tup match_frame; defines the size of the frame to be sliced
                for matching with the target.
        out: list of np arrays with LANDMARK_AMOUNT points along the ellipse
                around the center of each match.
        '''
        return [self._ellipse(m) for m in self.detector.match(image, self.search_region, match_frame)]

    def _ellipse(self, center, amount_of_points=LANDMARK_AMOUNT):
        '''
        Returns points along the ellipse around a center.
        '''
        ellipse = cv2.ellipse2Poly(tuple(center), (20, 20), 90, 0, 360, 4)
        print ("ellipse....: ",ellipse.shape)
        return Shape(np.hstack(ellipse[:amount_of_points, :].T))
    
   


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
        self.activeshape = ActiveShapeModel(self.pdmodel)

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
    



# def run()
        

# ------------- LOAD DATA -------------- #
loader = DataLoader()

training_set, test_set = loader.leave_one_out(test_index=5)



# training_set= loader.get_training()
# land=loader.get_land()

# --------------- TRAINING ---------------- #
training_images= training_set[0]
trainlandmarks = training_set[1]

# train a Feature Detection system
featuredetector = FDTraining()
# fully automatic:
# featuredetector.search_region = featuredetector.scan_region(trainlandmarks, diff=55, searchStep=20)
# semi-automatic:
# featuredetector.search_region = ((880, 1125), (1350, 1670), 20)
#print ('---Search space set to', featuredetector.search_region)
# print ('Done.')
# build and train an Active Shape Model

# pca1=Build_Model()



  
asm = ASMTraining(training_set, k=3, levels=3)
pca=asm.pdmodel
# ms = asm.get_model().get_mean().matrix
ms=pca.mean
# eigenVectors =pca_model.eigenvectors
eigenVectors =pca.eigenvectors



# eigenvalue=pca_model.eigenvalues
# mean_shape=pca_model.mean



# --------------- TESTING ----------------- #

testimage, testlandmarks = test_set


# remove some noise from the test image
testimage2 = remove_noise(testimage)
testimage1 =prep.calc_external_img_active_contour(testimage)

# perform feature matching to find init regions
# print( '---Searching for matches...')
# matches = featuredetector.match(testimage)
featuredetector.search_region = featuredetector.scan_region(trainlandmarks, diff=55, searchStep=20)
# print ('Done.')

# or perform manual initialisation (click on center)
matches = featuredetector._ellipse(plot.set_clicked_center(np.uint8(testimage)))


if not isinstance(testlandmarks, Shape):
    testlandmarks= Shape(testlandmarks)
    
meanShapeCentroid = (np.sum(ms.x)/30,np.sum(ms.y)/30)
centroid= tran.set_clicked_center(np.uint8(testimage))
matches1 = tran.initalizeShape(centroid, meanShapeCentroid, ms.matrix.T)
if not isinstance(matches1, Shape):
    matches1= Shape(matches1.T)


aligner= Aligner()
fitter=Fitter(pca)    
pose_para = aligner.get_pose_parameters(pca.mean, testlandmarks)
# set initial fitting pose parameters
fitter.start_pose = pose_para
# align model mean with region
points = aligner.transform(pca.mean, pose_para)
# meanShapeCentroid = (np.sum(testlandmarks.x)/30,np.sum(testlandmarks.y)/30)
plt.imshow(testimage)
plt.plot(points.x,points.y,'.')
plt.show()


plt.imshow(testimage)
plt.plot(testlandmarks.x,testlandmarks.y,'.')
plt.show()
# plt.plot(ms.x,ms.y,'.')
# plt.show()

# new_points,error = fit.fit_measure(points.matrix.T,10,testimage1)





def show_evolution(img, points_list):
    
    plt.figure()
    fig, ax = plt.subplots(figsize=(15, 7))
    n = len(points_list)
    hn = int(n/2)
    
    for i, landmark in enumerate(points_list):
        plt.subplot(2, hn, i+1)
        plt.imshow(img)
        plt.xticks(())
        plt.yticks(())   
        plt.plot(landmark[:,0], landmark[:,1], 'ro',  markersize=1)
     
    plt.show()

def active_shape(imageslice, init_shape, pca, length=10 ,MatchingON=True):
    'edge _img ,  pca_tooth : from prepratopn function '
    'tooth_point  ininitial position'

    # new_points, error = fit_measure(mean_shape, length, imageslice
    # new_fit = asm.activeshape.multiresolution_search(testimage, mean_shape, t=10, max_level=2, max_iter=10, n=0.2)
    new_points,error = fit.fit_measure(init_shape.matrix.T,length,imageslice)

    if(MatchingON):
        b, pose_param =fit.match_model_points(new_points, pca)
        x = fit.generate_model_point(b, pca)
        x= Shape(x)
       # x= np.vstack((x[0:29], x[30:59])).transpose()
        
        y = fit.inv_transform(x.matrix.T,pose_param)
        y= Shape(y.T)
    else:
        y = new_points
        
    return y,x


# y,x= active_shape(testimage1,matches[0].matrix.T,pca)
y,x= active_shape(testimage1,points,pca)

y= Shape([y.vector*np.linalg.norm(y.vector)])

v=fit.generate_model_point(np.zeros(len(pca.eigenvalues)),pca)
v= Shape(v)

print (pca.eigenvalues[0])
#new_fit = asm.activeshape.multiresolution_search(testimage, matches[0], t=10, max_level=0, max_iter=10, n=0.2)
plt.imshow(testimage)
plt.plot(y.x,y.y,'b')
plt.show()

# plt.imshow(testimage)

plt.plot(x.x,x.y,'r')



#for i in range(len(matches)):
# search and fit image
new_fit = asm.activeshape.multiresolution_search(testimage, matches[0], t=10, max_level=2, max_iter=10, n=0.2)
# Find the target that the new fit represents in order
# to compute the error. This is done by taking the smallest
# MSE of all targets.
testlandmarks= np.hstack(testlandmarks)
mse = np.zeros((testlandmarks.shape[0], 1))
mse = mean_squared_error(testlandmarks, new_fit)
best_fit_index = np.argmin(mse)
# implement maximally tolerable error
if int(mse) < MSE_THRESHOLD:
    print ('MSE:', mse)
    plot.render_shape_to_image(testimage, testlandmarks, color=(0, 0, 0))
else:
    print ('Bad fit. Needs to restart.')
    
    
# plt.plot(new_fit.x,new_fit.y)
# plt.show()

for i in range(len(points.lengthl)):
      # search and fit image
      new_fit = asm.activeshape.multiresolution_search(testimage, points, t=10, max_level=2, max_iter=10, n=0.2)
      # Find the target that the new fit represents in order
      # to compute the error. This is done by taking the smallest
      # MSE of all targets.
      # mse = np.zeros((trainlandmarks.shape[0], 1))
      # for i in range(mse.shape[0]):
      #     mse[i] = mean_squared_error(trainlandmarks[i], new_fit)
      # best_fit_index = np.argmin(mse)
      # # implement maximally tolerable error
      # if int(mse[best_fit_index]) < MSE_THRESHOLD:
      #     print ('MSE:', mse[best_fit_index])
      #     plot.render_shape_to_image(testimage, trainlandmarks[best_fit_index], color=(0, 0, 0))
      #     plt.imshow(testimage)
      #     plt.show()
      # else:
      #     print ('Bad fit. Needs to restart.')


# plt.plot(new_fit.y,new_fit.x,'.')
# plt.show()

# final_shape = Shape(land[best_fit_index,:])
# final_shape=final_shape.matrix

# plt.plot(final_shape[0,:],final_shape[1,:],'r')
# plt.show()


# np.sum((fit.vector - landmark)**2)/fit.length



# if __name__ == '__main__':
#     run()









