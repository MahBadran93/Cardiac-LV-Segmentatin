import os
import cv2
import numpy as np

from dentalvision.utils.structure import Shape
import loadnif as nif
import SimpleITK as sitk
import matplotlib.pyplot as plt 


# =============================================================================
# IMAGE_DIR = '../Project Data/_Data/Radiographs/'
# IMAGE_AMOUNT = 14
# IMAGE_DIM = (3023, 1597)
# 
# LANDMARK_DIR = '../Project Data/_Data/Landmarks/original/'
# LANDMARK_AMOUNT = 40            # amount of landmarks per tooth
# 
# =============================================================================

class DataLoader(object):
    '''
    This class provides methods to load specific landmark datasets
    for training and testing. It loads images and landmarks from
    directory paths specified in constants IMAGE_DIR and LANDMARK_DIR.
    '''
# =============================================================================
#     def __init__(self):
#         self.images = self._load_grayscale_images()
#         self.landmarks_per_image = self._load_landmarks_per_image()
# =============================================================================
        
    def __init__(self):
        self.images = self._load_grayscale_images()
        self.landmarks = self._load_landmarks()
        #self.training_set = [self.images, self.landmarks]
        self.training_set = [self.landmarks]
        
      
        
    def get_training(self):
        return self.training_set
    
    def get_land(self):
        return self.landmarks



    def leave_one_out(self, test_index=0):
        '''
        Divides into training and test sets by leaving one image and its
        landmarks out of the training set.

        in: int test_index; index to divide training/test
        out: np array images; array with grayscaled images per row
            np array landmarks; array with all landmarks as rows
            list of np arrays landmarks_per_image; array with rows of landmarks
                for each image
        '''
        training_images = np.delete(self.images,test_index,0)

        #training_images = np.asarray(self.images[:test_index] + self.images[:test_index+1])
        test_images = self.images[test_index]

        # create landmark training and test sets
        training_landmark = np.delete(self.landmarks,test_index,0)
        #training_landmark = np.asarray(self.landmarks[:test_index,:] + self.landmarks[:test_index+1,:])

        # training_landmarks_per_image = np.vstack((self.landmarks[:test_index], self.landmarks[test_index+1:]))

        # training_landmarks = np.vstack(training_landmarks_per_image[:][:])
        test_landmarks = self.landmarks[test_index,:]

        # compile training and test sets
        training_set = [training_images, training_landmark]
        test_set = [test_images, test_landmarks]

        return training_set, test_set
    
    
    
    # def leave_one_out(self, test_index=0):
    #     '''
    #     Divides into training and test sets by leaving one image and its
    #     landmarks out of the training set.

    #     in: int test_index; index to divide training/test
    #     out: np array images; array with grayscaled images per row
    #         np array landmarks; array with all landmarks as rows
    #         list of np arrays landmarks_per_image; array with rows of landmarks
    #             for each image
    #     '''
        
    #     #training_images = np.delete(self.images,test_index,0)

    #     #training_images = np.asarray(self.images[:test_index] + self.images[:test_index+1])
    #     test_images = self.images[test_index]
       
    #     # create landmark training and test sets
    #     training_landmark = np.delete(self.landmarks,test_index,0)
    #     #training_landmark = np.asarray(self.landmarks[:test_index,:] + self.landmarks[:test_index+1,:])

    #     # training_landmarks_per_image = np.vstack((self.landmarks[:test_index], self.landmarks[test_index+1:]))

    #     # training_landmarks = np.vstack(training_landmarks_per_image[:][:])
    #     test_landmarks = self.landmarks[test_index,:]


    #     return training_landmark, test_landmarks, test_images
    
    
    
    
    
    
# =============================================================================
#     def _load_grayscale_images(self):
#         '''
#         Load the images dataset.
#         '''
#         images = []
#         for i in os.listdir(IMAGE_DIR):
#             if i.endswith('.tif'):
#                 path = IMAGE_DIR + i
#                 images.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))
#         return images
#     
# =============================================================================
    

    def _load_grayscale_images(self):
        
      '''
      load the the image from saveDataSet.py file using simpleITK library 
      '''
      images = []
      path = '../training/'
      count = 0
      for root, dirs, files in os.walk(path): # 100 iteration, num of patients in training Folder
          dirs.sort()
          files.sort()
          for name in files: # iterate 6 times, depends on num of files 
              simpitkImgSys = nif.loadNiftSimpleITK(root,files[2:3].pop())
              simpitkImgDia = nif.loadNiftSimpleITK(root,files[4:5].pop())
              sliceGT1 = nif.loadNiftSimpleITK(root,files[3:4].pop())
              sliceGT2 = nif.loadNiftSimpleITK(root, files[5:6].pop())
              # itereate depends on num of slices
              for i in range (simpitkImgSys.GetSize()[2]): #sliceGT1.shape[2]
                  count = count + 1
                  #SaveToFolder(simpitkImgSys, i, count)
                  imageslice = sitk.GetArrayFromImage(simpitkImgSys[:,:,i])
                  imgGT1 = sitk.GetArrayFromImage(sliceGT1[:,:,i])
                  imgGT1[:,:][imgGT1[:,:] != 3] = 0
                  slice1Copy = np.uint8(imgGT1[:,:])
                  imgGT1 = cv2.Canny(slice1Copy,0,1)
                  listOfIndex = np.argwhere(imgGT1 !=0)
    
                  #print(np.mean(imgGT1))
                  if (len(listOfIndex) == 0) :
                      continue
                  images.append(imageslice)
                  # break
              for i in range (simpitkImgDia.GetSize()[2]): 
                  count = count + 1
                  imgGT2 = sitk.GetArrayFromImage(sliceGT2[:,:,i])
                  imgGT2[:,:][imgGT2[:,:] != 3] = 0
                  slice2Copy = np.uint8(imgGT2[:,:])
                  imgGT2 = cv2.Canny(slice2Copy,0,1)
                  listOfIndex = np.argwhere(imgGT2 !=0)
    
                  #SaveToFolder(simpitkImgDia,i,count)
                  if (len(listOfIndex)) == 0:
                      continue
                  images.append(sitk.GetArrayFromImage(simpitkImgDia[:,:,i]))
    
                  
              break
      
      return np.asarray(images)

# =============================================================================
#     def _load_landmarks_per_image(self):
#         '''
#         Compile landmarks per image for convenience in grayscale level
#         training. This training phase needs an accurate relation between
#         the images and their corresponding landmarks.
# 
#         Needs to be run after _load_grayscale_images()!
#         '''
#         if not self.images:
#             raise IOError('Images have not been loaded yet.')
# 
#         landmarks_per_image = []
#         for i in range(len(self.images)):
#             # search for landmarks that include reference to image in path
#             lms = [self._parse(LANDMARK_DIR + s) for s in os.listdir(LANDMARK_DIR) if ('landmarks'+str(i+1)+'-2.txt') in s]
#             lms2 = [self._parse(LANDMARK_DIR + s) for s in os.listdir(LANDMARK_DIR) if ('landmarks'+str(i+1)+'-3.txt') in s]
#             l = lms[0].merge(lms2[0])
#             landmarks_per_image.append([l.vector])
# 
#         return np.asarray(landmarks_per_image)
# =============================================================================
    
    def _load_landmarks(self):
        '''
        Compile landmarks per image for convenience in grayscale level
        training. This training phase needs an accurate relation between
        the images and their corresponding landmarks.

        Needs to be run after _load_grayscale_images()!
        '''
    
        self.landmarks = np.genfromtxt("NotAlignedlandmark.csv", delimiter=",")
        print ("load image done...")
        return self.landmarks
        

    
    

# =============================================================================
#     def _parse(self, path):
#         '''
#         Parse the data from path directory and return arrays of x and y coordinates
#         Data should be in the form (x1, y1)
# 
#         in: String pathdirectory with list of landmarks (x1, y1, ..., xN, yN)
#         out: 1xc array (x1, ..., xN, y1, ..., yN)
#         '''
#         data = np.loadtxt(path)
#         x = np.absolute(data[::2, ])
#         y = data[1::2, ]
#         return Shape(np.hstack((x, y)))
# =============================================================================

