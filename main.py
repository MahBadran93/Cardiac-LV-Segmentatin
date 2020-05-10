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
from dentalvision.utils import plot
import loadnif as nif
import matplotlib.pyplot as plt
import translateinistialshape as tran
import  sklearn.metrics as sk
import random
import scipy
from scipy import ndimage
import math
from AutomateLandMarks import LandMarks








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
        ellipse = cv2.ellipse2Poly(tuple(center), (80, 210), 90, 0, 360, 4)
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

        # 2. Train GRAYSCALE MODELs using multi-resolution images
        print ('---Training Gray-Level Model pyramid...')
        glmodel_pyramid = self.graylevelmodel_pyramid(k=k, levels=levels)

        # 3. Train ACTIVE SHAPE MODEL
        print ('---Initialising Active Shape Model...')
        self.activeshape = ActiveShapeModel(self.pdmodel, glmodel_pyramid)

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
        
proc= LandMarks()

# ------------- LOAD DATA -------------- #
loader = DataLoader()

training_set, test_set = loader.leave_one_out(test_index=9)

# training_set= loader.get_training()
# land=loader.get_land()

# --------------- TRAINING ---------------- #
training_images= training_set[0]
trainlandmarks = training_set[1]

print (training_images[0].shape)
# train a Feature Detection system
featuredetector = FDTraining()
# fully automatic:
featuredetector.search_region = featuredetector.scan_region(trainlandmarks, diff=55, searchStep=20)
# semi-automatic:
# featuredetector.search_region = ((880, 1125), (1350, 1670), 20)
#print ('---Search space set to', featuredetector.search_region)
# print ('Done.')

# build and train an Active Shape Model
  
asm = ASMTraining(training_set, k=3, levels=3)
pca_model=asm.pdmodel
ms = asm.get_model().get_mean().matrix

eigenVectors =pca_model.eigenvectors
eigenvalue=pca_model.eigenvalues
mean_shape=pca_model.mean


# --------------- TESTING ----------------- #

testimage, testlandmarks = test_set


# remove some noise from the test image

testimage = remove_noise(testimage)
plt.imshow(testimage)
plt.show()

# perform feature matching to find init regions
# print( '---Searching for matches...')
# matches = featuredetector.match(testimage)
# print ('Done.')

# or perform manual initialisation (click on center)
matches = [featuredetector._ellipse(plot.set_clicked_center(np.uint8(testimage)))]

#testlandmarks= land[2,:]

# plt.imshow(testimage)
# x=testlandmarks[0:29]
# y= testlandmarks[30:59]

# testlandmarks1 = Shape(testlandmarks)
# testlandmarks=testlandmarks1.matrix



# plt.plot(testlandmarks[0,:],testlandmarks[1,:],'r')
# plt.show()




# meanShapeCentroid = (np.sum(ms[:,0])/30,np.sum(ms[:,1])/30)
# centroid= tran.set_clicked_center(np.uint8(testimage))
# matches1 = tran.initalizeShape(centroid, meanShapeCentroid, ms.T)



# x, y = np.hsplit(matches1,2)
# length = x.size
# vector = np.vstack((x, y))
# matrix = np.hstack((x, y))
# plt.plot(matches1[:,0],matches1[:,1])
# plt.show()


# if not isinstance(matches1, Shape):
#     matches= [Shape(matches1)]


print(len(matches))
print (matches[0].x)
print ("matches: ", matches[0])
plt.plot(matches[0].y,matches[0].x)
plt.show()



# #for i in range(len(matches)):
# # search and fit image
# new_fit = asm.activeshape.multiresolution_search(testimage, matches[0], t=10, max_level=2, max_iter=10, n=0.2)
# # Find the target that the new fit represents in order
# # to compute the error. This is done by taking the smallest
# # MSE of all targets.
# testlandmarks= np.hstack(testlandmarks)
# mse = np.zeros((testlandmarks.shape[0], 1))
# mse = mean_squared_error(testlandmarks, new_fit)
# best_fit_index = np.argmin(mse)
# # implement maximally tolerable error
# if int(mse) < MSE_THRESHOLD:
#     print ('MSE:', mse)
#     plot.render_shape_to_image(testimage, testlandmarks, color=(0, 0, 0))
# else:
#     print ('Bad fit. Needs to restart.')
    
    
# plt.plot(new_fit.x,new_fit.y)
# plt.show()

# print ("test landmark : ", trainlandmarks[0])
# for i in range(len(matches)):
#      # search and fit image
#      new_fit = asm.activeshape.multiresolution_search(testimage, matches[i], t=10, max_level=2, max_iter=10, n=0.2)
#      # Find the target that the new fit represents in order
#      # to compute the error. This is done by taking the smallest
#      # MSE of all targets.
#      # mse = np.zeros((trainlandmarks.shape[0], 1))
#      # for i in range(mse.shape[0]):
#      #     mse[i] = mean_squared_error(trainlandmarks[i], new_fit)
#      # best_fit_index = np.argmin(mse)
#      # # implement maximally tolerable error
#      # if int(mse[best_fit_index]) < MSE_THRESHOLD:
#      #     print ('MSE:', mse[best_fit_index])
#      #     plot.render_shape_to_image(testimage, trainlandmarks[best_fit_index], color=(0, 0, 0))
#      #     plt.imshow(testimage)
#      #     plt.show()
#      # else:
#      #     print ('Bad fit. Needs to restart.')


# plt.plot(new_fit.y,new_fit.x,'.')
# plt.show()

# final_shape = Shape(land[best_fit_index,:])
# final_shape=final_shape.matrix

# plt.plot(final_shape[0,:],final_shape[1,:],'r')
# plt.show()


# np.sum((fit.vector - landmark)**2)/fit.length



# if __name__ == '__main__':
#     run()

pose_param2={}

def strongest_edge_point_on_normal(a,b,c,length, edge_img):
    
    print (a,b,c )
    
    rad = get_normal_angle(a,b,c)
    points = get_points_on_angle(b, rad, length)
    edge_strength = edge_strength_at_points(points, edge_img)
    id_edge_point = np.argmax(edge_strength)
    edge_point = points[id_edge_point]

    return edge_point 

def get_points_on_angle(point, rad, length):
    print (point,rad,length)
    
    
    points = np.empty((2*length+1, 2))
    points[0] = point
    for i, x in enumerate(range(1,length+1)):

        points[2*i+1] = get_point_at_distance(point, x, rad)
        points[2*i+2] = get_point_at_distance(point, -x, rad)
        
    return points

def get_point_at_distance(point, dist, rad):
    new_point = np.zeros_like(point)
    y = int(np.around(math.sin(rad) * dist))
    x = int(np.around(math.cos(rad) * dist))
    new_point[0] = point[0] + x
    new_point[1] = point[1] + y
    return new_point

def edge_strength_at_points(points ,edge_img):
    
    gradient = np.empty(len(points))
    for i, p in enumerate(points):
        gradient[i] = edge_img[int(p[1]),int(p[0])]
        
    return gradient


def error_measure(p1, p2):
    
    x1, y1 = p1
    x2, y2 = p2
    #dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return math.hypot(x2 - x1, y2 - y1)

def get_normal_angle(a,b,c):
    print (a,b,c )

    
    if( is_horizontal(a,b,c) ):
        return math.pi/2
    if( is_vertical(a,b,c) ):
        return 0
    
        
def is_horizontal(a,b,c):
    if(a[1] == b[1] and b[1] == c[1]):
        return True
    return False
    
def is_vertical(a,b,c):
    if(a[0] == b[0] and b[0] == c[0]):
        return True
    return False

# def get_range_of(i, pca):
#     bound = 3*math.sqrt(pca.eigenvalue)
#     return bound

def get_range_of(i, pca):
    eigenvalues = pca.eigenvalues
    bound = 3*math.sqrt(eigenvalues[i])
    return bound


def generate_model_point(b,pca):
    eigenvectors = pca.eigenvalues
    P = eigenvectors
    #b = get_eigenvalues(pca)
    xm = pca.mean.vector
    
    x =  np.dot(P,b)
    return x + xm

def get_eigenvalues(pca):
    
    return pca.explained_variance_

def get_eigenvectors(pca):
    return 

def get_mean(pca):
    return 




def get_pose_param_for_transformation(A,B):
    
    
    
    s = round(len(A)/3)
    r = random.randint(0, s) 
    r2 , r3 = round(r+s), round(r+2*s)
    
    pts1 = np.float32((A[r],A[r2],A[r3]))
    pts2 = np.float32((B[r],B[r2],B[r3]))
    M = cv2.getAffineTransform(pts1,pts2)
    
    ax, ay, tx, ty = M[0,0], M[1,0], M[0,2], M[1,2]
    return (ax, ay, tx, ty)

def tarekfunctionforposeparameter(A,B):
      pose_matrix,disparity,pose_param= proc.procrustes(A,B)
      
      return pose_param
    


def inv_transform(p, pose_param):
    (ax, ay, tx, ty) = pose_param
    scale_rotation = np.array([[ax, ay],[ -ay, ax]])
    divisor = 1 / (ax**2 + ay**2)
    Txy = np.array([tx, ty])
    transformed = np.dot(scale_rotation*divisor, np.transpose(p-Txy))
    return transformed.transpose()

def transform(p, pose_param):
    (ax, ay, tx, ty) = pose_param
    scale_rotation = np.array([[ax, -ay],[ ay, ax]])
    Txy = np.array([tx, ty])
    
    p_transformed = np.dot(scale_rotation, p.transpose()) 
    return (np.add(Txy, p_transformed.transpose()))

def project_to_tangent_plane(y, pca):
    xm = pca.mean.vector
    y = y.reshape(-1)
    return y / np.dot(y,xm)

def update_model_param(y, pca):
    xm = pca_model.mean.vector
    PT = pca_model.eigenvectors.T
    return np.dot(PT, y.reshape(-1) - xm)

def constraint_model_param(b,pca):
    
    for i in range(len(b)):
        bound = get_range_of(i,pca)
        if( b[i] > bound):
            b[i] = bound
        elif ( b[i] < -bound):
            b[i] = -bound
    return b


def match_model_points(Y, pca):
    b = np.zeros(len(pca.eigenvalues)) 

    max_conv_iter = 20
    best_b = b
    best_pose_param = (0,0,0,0)
    best_MSE = np.inf
    convergence_iter = max_conv_iter

    while(1):
        
        x = generate_model_point(b, pca)
        # if not isinstance(x, Shape):
        #     x= [Shape(x.T)]
      
        # x = np.vstack((x[0:29], x[30:59])).transpose()
      
        x5,y5 = np.split(np.hstack(x), 2)
        x = np.vstack((x5, y5)).T 


        # pose_param = get_pose_param_for_transformation(Y,x)
        pose_param=tarekfunctionforposeparameter(Y.matrix.T,x)
        
        global pose_param2
        pose_param2=pose_param
    
        pose_param= pose_param2['rotation'][0,0],pose_param2['rotation'][1,0],pose_param2['translation'][0],pose_param2['translation'][1]
        
        
     
        Y_pred = inv_transform(x,pose_param)
        # print(Y_pred.shape)
        # plt.plot(Y_)
        
        MSE = sk.mean_squared_error(Y.matrix.T, Y_pred)
        
        if(MSE < best_MSE):
            best_b = b
            best_pose_param = pose_param
            best_MSE = MSE
            convergence_iter = max_conv_iter
        
        convergence_iter -= 1
        if(convergence_iter == 0 or best_MSE < 1):
            #print(convergence_iter, best_MSE)
            break;    
          
        y = transform(Y.matrix.T,pose_param)
        
        y = project_to_tangent_plane(y, pca)
        x,y = np.split(np.hstack(y), 2)
        y = np.vstack((x, y))
       
        # y = np.vstack((y[0:29], y[30:59])).transpose()

        b = update_model_param(y, pca)
        b = constraint_model_param(b,pca)
        
        print (b)

    return best_b, best_pose_param;





def active_shape(imageslice, mean_shape, pca_result, length=10 ,MatchingON=True):
    'edge _img ,  pca_tooth : from prepratopn function '
    'tooth_point  ininitial position'

    # new_points, error = fit_measure(mean_shape, length, imageslice
    new_fit = asm.activeshape.multiresolution_search(testimage, mean_shape, t=10, max_level=2, max_iter=10, n=0.2)

    new_points =new_fit
    if(MatchingON):
        b, pose_param = match_model_points(new_points, pca_model)
        x = generate_model_point(b, pca_model)
        x= np.vstack((x[0:29], x[30:59])).transpose()
        y = inv_transform(x,pose_param)
    else:
        y = new_points
        
    return y



y= active_shape(testimage,matches[0],pca_model)
#new_fit = asm.activeshape.multiresolution_search(testimage, matches[0], t=10, max_level=0, max_iter=10, n=0.2)
plt.imshow(testimage)
plt.plot(y[:,0],y[:,1],'r.')
plt.show()