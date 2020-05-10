from AutomateLandMarks import  LandMarks
import saveDataSet as saveData
import loadnif as nif
import numpy as np
import matplotlib.pyplot as plt
import cv2
import  shapely as shp
import shapely.geometry as gmt
import pylab as pl
import  scipy as sc
import  SimpleITK as sitk
from  model import Build_Model
from tempfile import TemporaryFile
 
import math

from skimage.color import rgb2gray
from sklearn.decomposition import PCA
import matplotlib
from skimage.color import rgb2gray
from skimage.filters import gaussian
import scipy
from scipy import ndimage

from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage.morphology import disk
from skimage.filters import rank
from skimage import exposure

from skimage.filters import roberts, sobel, scharr, prewitt
from skimage import feature, img_as_float

from dentalvision.utils import structure
from dentalvision.utils import plot
from dentalvision.utils.structure import Shape

import main 



# centroid = (np.sum(ms[:,0])/ms.shape[0],np.sum(ms[:,1])/ms.shape[0])
# initShape = fm.initalizeShape(click,centroid,ms)
# plt.plot(initShape[:,0],initShape[:,1],'.')

# plt.imshow(imageT)

# def set_clicked_center(img):
     
     
     
#      '''
#      Show image and register the coordinates of a click into
#      a global variable.
#      '''
#      def detect_click(event, x, y, flags, param):
#          global click
#          click = (x, y)
     
#      cv2.namedWindow("clicked", cv2.WINDOW_NORMAL)
#      cv2.setMouseCallback("clicked", detect_click)
     
#      while True:
#         cv2.imshow("clicked", img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         if click:
#             return click

# def initalizeShape(centroid,meanShapeCentroid, modelShape):
#     # centroid : center coords , i.e : (120,120).
#     # modelShape : the shape to center around centroid. 
#     subtractValue = np.subtract(np.array(centroid),np.array(meanShapeCentroid))
#     translatedShpe = np.add(modelShape,subtractValue)
    
#     return translatedShpe # return a translated shape around a specific fixed centroid for all sh



# =============================================================================
# LANDMARK_AMOUNT = 30            # amount of landmarks per tooth
# MSE_THRESHOLD = 60000            # maximally tolerable error
# 
# 
# 
# =============================================================================
#.............................Testing.....................................................


# =============================================================================
# path = '../training/patient001/patient001_frame01.nii.gz'
# image = nif.loadSimpleITK(path)
# image2 = nif.loadNifti(path)
# 
# imageslice =nif.getSliceITK(image,2)
# 
# click=()
# 
# def _ellipse(center, amount_of_points=LANDMARK_AMOUNT):
#         '''
#         Returns points along the ellipse around a center.
#         '''
#         ellipse = cv2.ellipse2Poly(tuple(center), (80, 210), 90, 0, 360, 4)
#         return Shape(np.hstack(ellipse[:amount_of_points, :].T))
#     
#     
# def remove_noise(img):
#     '''
#     Blur image to partially remove noise. Uses a median filter.
#     '''
#     return cv2.medianBlur(img, 5)



# ........... Create Training Model...........................
# # =============================================================================
# showLand = LandMarks()
# landMarkedShapes,originalShape, sampledwithoutAlighn= showLand.GenerateSampleShapeList()

# # convert the list to numpy array 
# landMarkedShapesR = np.stack(landMarkedShapes,axis=0)
# landMarkedShapesR2 = np.array(landMarkedShapes).T

# # final Matrix of Sampled & aligned shapes, Diminsion (2,30,1788) 
# LandMarkFinalMatrix =landMarkedShapesR.T

# # Reshape the final matrix to Diminsion (2,1788, 30)
# AlignedMAtrixOfShapes = np.transpose(LandMarkFinalMatrix,(0,2,1)) 
# # =============================================================================

# [numberOfMode,pca_model,Landmarkcoulmn]= Build_Model()


    
    
# testlandmarks=pca_model.mean_

# testimage = remove_noise(imageslice)


# matches = [_ellipse(plot.set_clicked_center(np.uint8(testimage)))]
# #matches = [_ellipse((107,126))]

# print(len(matches))

# plt.plot(matches[0].x,matches[0].y)
# plt.show()


# =============================================================================


# model_final=[evali,eveci,mean_shape,mode]
# for i in range (len(landMarkedShapes)):
#     x1 = [p[0] for p in landMarkedShapes[i]]
#     y1 = [p[1] for p in landMarkedShapes[i]]
#     plt.plot(x1,y1,'.')
#     plt.show()




# print (pca_result.get_covariance().shape)



# x= eveci[1,:]
    
# xxx=evali[1]
    
    
# b = np.dot(x,xxx)
# y = mean_shape + b

# #plt.axis([-216, 304, -216, 304])
# plt.plot(y[0:29],y[30:59])
# plt.show()

# A1=mean_shape-y
# A = A1.reshape(30,2)
# plt.plot(A[:,0],A[:,1])
# plt.show()

# test= np.linalg.svd(A)

# mean_shape=ms = np.vstack((mean_shape[0:29], mean_shape[30:59])).transpose()




def calc_external_img_active_contour(img): 

  #  median = median_filter(img)
    contrast = contrast_stretching(img)
    ext_img = canny(contrast)

    return ext_img

def median_filter(img, size = 3):
    return scipy.signal.medfilt(img, size).astype(np.uint8)

def contrast_stretching(img):
    # Contrast stretching
    #p2, p98 = np.percentile(img, (0, 20))
    return exposure.rescale_intensity(img, in_range=(0.05*255, 0.6*255))

def canny(img):
    return feature.canny(img, sigma=2)


def preperation(radiograph):#, tooth_variations):
    'this is imortant to get edge image and pca_tooth for '  
    "tooth_variations: in oue case this landmarkcoulm 1780*60 "

    
    #median = prep.median_filter(radiograph)
#     edge_img = prep.edge_detection_low(median)

    edge_img = calc_external_img_active_contour(radiograph)
    # pca_tooth = PCA.PCA_analysis(tooth_variations, None)
    
    return edge_img   #, pca_tooth


def fit_measure(points, length, edge_img):
    "point: touth point  initial shape "
    
    size = len(points)
    new_points = np.empty((size,2))
    total_error = 0
    
    for i in range(size):
        #print(i)
        if(i==size-1):
            p1, p2, p3 = points[i-1], points[i], points[0] 
        else:
            p1, p2, p3 = points[i-1], points[i], points[i+1]

        p2_new = strongest_edge_point_on_normal(p1, p2, p3 ,length, edge_img)
        #print(p2_new)
        total_error += error_measure(p2, p2_new)
        new_points[i] = p2_new
        
    return new_points, total_error;  
    
# def _compute_normal(points):
#         '''
#         Compute the normal between three points.
#         '''
#         prev, curr, nex = points
#         return self._normal(prev, nex)

#     def _normal(self, a, b):
        
#         '''
#         Compute the normal between two points a and b.

#         in: tuple coordinates a and b
#         out: 1x2 array normal
#         '''
#         d = b - a
#         tx, ty = d/math.sqrt(np.sum(np.power(d, 2)))
#         return np.array([-1*ty, tx])

# def _sample(self, starting_point):
#         '''
#         Returns 2k+1 points along the normal
#         '''
#         positives = []
#         negatives = []
#         start = [(int(starting_point[0]), int(starting_point[1]))]

#         i = 1
#         while len(positives) < self.k:
#             new = (starting_point[0] - i*self.normal[0], starting_point[1] - i*self.normal[1])
#             if (new not in positives) and (new not in start):
#                 positives.append(new)
#             i += 1

#         i = 1
#         while len(negatives) < self.k:
#             new = (starting_point[0] + i*self.normal[0], starting_point[1] + i*self.normal[1])
#             if (new not in negatives) and (new not in start):
#                 negatives.append(new)
#             i += 1

#         negatives.reverse()

#         return np.array(negatives + start + positives)
        
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

def get_range_of(i, pca):
    bound = 3*math.sqrt(evali[i])
    return bound

def generate_model_point(b):
    eigenvectors = eveci
    P = eigenvectors.transpose()
    #b = get_eigenvalues(pca)
    xm = mean_shape
    
    x =  np.dot(P,b)
    return x + xm

def get_eigenvalues(pca):
    
    return pca.explained_variance_

def get_eigenvectors(pca):
    return 

def get_mean(pca):
    return 

def generate_model_point(b, pca):
    eigenvectors = pca.components_
    P = eigenvectors.transpose()
    #b = get_eigenvalues(pca)
    xm = pca.mean_
    
    x =  np.dot(P,b)
    return x + xm


def get_pose_param_for_transformation(A,B):
    
    s = round(len(A)/3)
    r = random.randint(0, s) 
    r2 , r3 = round(r+s), round(r+2*s)
    
    pts1 = np.float32((A[r],A[r2],A[r3]))
    pts2 = np.float32((B[r],B[r2],B[r3]))
    M = cv2.getAffineTransform(pts1,pts2)
    
    ax, ay, tx, ty = M[0,0], M[1,0], M[0,2], M[1,2]
    return (ax, ay, tx, ty)


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
    xm = pca.mean_
    y = y.reshape(-1)
    return y / np.dot(y,xm)

def update_model_param(y, pca):
    xm = pca.mean_
    PT = pca_result.get_eigenvectors(pca)
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
    b = np.zeros(len(pca.components_)) 

    max_conv_iter = 20
    best_b = b
    best_pose_param = (0,0,0,0)
    best_MSE = np.inf
    convergence_iter = max_conv_iter

    while(1):
        
        x = generate_model_point(b, pca)
        x = np.vstack((x[0:29], x[30:59])).transpose()
        pose_param = get_pose_param_for_transformation(Y,x)
        
        Y_pred = inv_transform(x,pose_param)
        MSE = sklearn.metrics.mean_squared_error(Y, Y_pred)
        
        if(MSE < best_MSE):
            best_b = b
            best_pose_param = pose_param
            best_MSE = MSE
            convergence_iter = max_conv_iter
        
        convergence_iter -= 1
        if(convergence_iter == 0 or best_MSE < 1):
            #print(convergence_iter, best_MSE)
            break;    
          
        y = transform(Y,pose_param)
        
        y = project_to_tangent_plane(y, pca)
        y = np.vstack((y[0:29], y[30:59])).transpose()

        b = update_model_param(y, pca)
        b = constraint_model_param(b,pca)

    return best_b, best_pose_param;



def active_shape(imageslice, mean_shape, pca_result, length=10 ,MatchingON=True):
    'edge _img ,  pca_tooth : from prepratopn function '
    'tooth_point  ininitial position'

    new_points, error = fit_measure(mean_shape, length, imageslice)     
    
    if(MatchingON):
        b, pose_param = match_model_points(new_points, pca_result)
        x = generate_model_point(b, pca_result)
        x= np.vstack((x[0:29], x[30:59])).transpose()
        y = inv_transform(x,pose_param)
    else:
        y = new_points
        
    return y



img= preperation(imageslice)

plt.imshow(img)

plt.imshow(imageslice)

y= active_shape(img,mean_shape,pca_result)


click=()

def set_clicked_center(img):
    '''
    Show image and register the coordinates of a click into
    a global variable.
    '''
    def detect_click(event, x, y, flags, param):
        global click
        click = (x, y)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.namedWindow("clicked", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("clicked", detect_click)

    while True:
        cv2.imshow("clicked", img)
      
        if click:
            print(click)
            return click
        
def _ellipse(center, amount_of_points=30):
    '''
    Returns points along the ellipse around a center.
    '''
    ellipse = cv2.ellipse2Poly(tuple(center), (80, 210), 90, 0, 360, 4)
    return Shape(np.hstack(ellipse[:amount_of_points, :].T))   

center=()
center= set_clicked_center(np.uint8(imageslice))

'''

# plt.axis([-216, 304, -216, 304])
# plt.plot(y[0:29],y[30:59],".")
# plt.show()

class Point ( object ):
  """ Class to represent a point in 2d cartesian space """
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __add__(self, p):
    """ Return a new point which is equal to this point added to p
    :param p: The other point
    """
    return Point(self.x + p.x, self.y + p.y)

  def __div__(self, i):
    return Point(self.x/i, self.y/i)

  def __eq__(self, other):
    return self.x == other.x and self.y == other.y

  def __ne__(self, other):
    return not self.__eq__(other)

  def __repr__(self):
    """return a string representation of this point. """
    return '(%f, %f)' % (self.x, self.y)

  def dist(self, p):
    """ Return the distance of this point to another point

    :param p: The other point
    """
    return math.sqrt((p.x - self.x)**2 + (p.y - self.y)**2)

   
# # path = '../training/patient001/patient001_frame01.nii.gz'
# # image = nif.loadSimpleITK(path)
# # image2 = nif.loadNifti(path)
# # 
# # imageslice =nif.getSliceITK(image,2)
# # 
# # # imageslice2 = nif.getSlice(image2,8) 
# # 
# # # x1 = [p[0] for p in originalShape[7]]
# # # y1 = [p[1] for p in originalShape[7]]
# # # x2 = [p[0] for p in sampledwithoutAlighn[6]]
# # # y2 = [p[1] for p in sampledwithoutAlighn[6]]
# # # x3 = [p[0] for p in landMarkedShapes[6]]
# # # y3 = [p[1] for p in landMarkedShapes[6]]
# # 
# # size = imageslice.shape
# # 
# # plt.imshow(imageslice)
# # # plt.plot(x1,y1,'.')
# # # plt.plot(x2,y2)
# # 
# # # plt.plot(x3,y3)
# # 
# # # plt.show()
# # 
# # 
# # 
# # 
# # def __init__(model_final, imageslice, t=Point(0.0,0.0)):
# #     image = imageslice
# #     g_image = []
# #     for i in range(0,4):
# #       g_image.append(__produce_gradient_image(image, 2**i))
# #       
# #     plt.imshow(g_image[0])
# #     plt.show()
# # 
# #     plt.imshow(g_image[1])
# #     plt.show()
# # 
# #     plt.imshow(g_image[2])
# #     plt.show()
# # 
# #     plt.imshow(g_image[3])
# #     
# #     plt.show()
# #     model_final = model_final
# #       # Copy mean shape as starting shape and transform it to origin
# #     # shape = Shape.from_vector(model_final.mean_shape).transform(t)
# #     # # And resize shape to fit image if required
# #     # if __shape_outside_image(self.shape, self.image):
# #     #   self.shape = self.__resize_shape_to_fit_image(self.shape, self.image)
# #     
# #     
# #         
# # 
# # def __produce_gradient_image(i, scale):
# #     size =i.shape
# # #    grayImage = rgb2gray(i)
# #     size = [s/scale for s in size]
# #     grey_image_small =  cv2.resize(i , (int(size[0]),int(size[1])))
# # 
# #     # Output dtype = cv2.CV_8U
# #     sobelx8u = cv2.Sobel(grey_image_small,cv2.CV_8U,1,0,ksize=5)
# # 
# #     # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
# #     sobelx64f = cv2.Sobel(grey_image_small,cv2.CV_64F,1,0,ksize=5)
# #     abs_sobel64f = np.absolute(sobelx64f)
# #     sobel_8u = np.uint8(abs_sobel64f)
# #     
# #     # plt.subplot(1,3,1),plt.imshow(grey_image_small,cmap = 'gray')
# #     # plt.title('Original'), plt.xticks([]), plt.yticks([])
# #     
# #     # plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
# #     # plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
# #     
# #     # plt.show()
# #     return sobel_8u
# # 
# # 
# # __init__(model_final, imageslice)      
# #         
# # '''
# # 
# #     
# # 
# # '''
# # # Plot the list of all the shapes 
# # for i in range (len(landMarkedShapes)):
# #     plt.axis([-0.4, 0.6, -0.4, 0.6])
# #     x1 = [p[0] for p in landMarkedShapes[i]]
# #     y1 = [p[1] for p in landMarkedShapes[i]]
# #     plt.plot(x1,y1)
# # '''
# # 
# # '''
# # # plot the sampled and aligned list of shapes (Numpy array)
# # for i in range(AlignedMAtrixOfShapes.shape[1]):
# #     x1 = AlignedMAtrixOfShapes[0,i,:]
# #     y1 = AlignedMAtrixOfShapes[1,i,:]
# #     plt.axis([-0.4, 0.6, -0.4, 0.6])
# #     plt.plot(x1,y1)
# #     plt.show()
# # '''    
# # 
# # 
# # =============================================================================
