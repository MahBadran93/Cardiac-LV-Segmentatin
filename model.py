
from AutomateLandMarks import  LandMarks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dentalvision.utils.structure import Shape
from dentalvision.pdm.gpa import gpa
from dentalvision.utils.plot import plot


 
# def create_pdm(shapes):
#     '''
#     Create a new point distribution model based on landmark data.

#     Step 1: Generalised Procrustes Analysis on the landmark data
#     Step 2: Principal Component Analysis on the GPAed landmark data
#             This process will return an amount of eigenvectors from
#             which we construct deviations from the mean image.
#     Step 3: Create a deformable model from the processed data

#     In: list of directories of the landmark data
#     Out: DeformableModel instance created with preprocessed data.
#     '''
#     # perform gpa
#     mean, aligned = gpa(np.asarray(shapes))
#     plot('gpa', mean, aligned)

#     # perform PCA
#     eigenvalues, eigenvectors, m = pca(aligned, mean=mean, max_variance=0.99)
#     plot('eigenvectors', mean, eigenvectors)

#     # create PointDistributionModel instance
#     model = PointDistributionModel(eigenvalues, eigenvectors, mean)
#     plot('deformablemodel', model)

#     return model


class PointDistributionModel(object):
    '''
    Model created based on a mean image and a matrix
    of eigenvectors and corresponding eigenvalues.
    Based on shape parameters, it is able to create a
    variation on the mean shape.

    Eigenvectors are scaled according to Blanz p.2 eq.7.
    '''
    def __init__(self, eigenvalues, eigenvectors, mean):
        self.dimension = eigenvalues.size
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.mean = Shape(mean)
        

        # create a set of scaled eigenvectors
        self.scaled_eigenvectors = np.dot(self.eigenvectors, np.diag(self.eigenvalues))
        
    def get_mean(self):
        return self.mean
    

    def deform(self, shape_param):
        '''
        Reconstruct a shape based on principal components and a set of
        parameters that define a deformable model (see Cootes p. 6 eq. 2)

        in: Tx1 vector deformable model b
        out: 1xC deformed image
        '''
        return Shape(self.mean.vector + self.scaled_eigenvectors.dot(shape_param))
    
def Build_Model():
    
    
    # retrun landmark for all shape and save it to CSV file 
    # showLand = LandMarks()
    # landMarkedShapes,originalShape, sampledwithoutAlighn= showLand.GenerateSampleShapeList()
    # np.savetxt("alllandmark.csv" , landMarkedShapes, delimiter=",")
    # print ("saved done... ")
    
    # load landmark from alllandmark.Csv file 
    Landmarkcoulmn= np.genfromtxt("alllandmark.csv", delimiter=",")
    
    pca = PCA()
    pca_model = pca.fit(Landmarkcoulmn)
    eigenVectors =pca_model.components_
    eigenvalue=pca_model.explained_variance_
    mean=pca_model.mean_
    
    model = PointDistributionModel(eigenvalue, eigenVectors, mean)
    
    #mean, aligned = gpa(np.asarray(Landmarkcoulmn))
    plot('gpa', mean, Landmarkcoulmn)

    # perform PCA
 #   eigenvalues, eigenvectors, m = pca(aligned, mean=mean, max_variance=0.99)
    plot('eigenvectors', mean, eigenVectors)

    # create PointDistributionModel instance
  #  model = PointDistributionModel(eigenvalues, eigenvectors, mean)
    plot('deformablemodel', model)


    
    
    # Find number of modes(eigenvalue) required to describe the most important variance of the data 
    t = 0
    for i in range(len(eigenvalue)):
      if sum(eigenvalue[:i]) / sum(eigenvalue) < 0.99:
          t = t + 1
      else: break
  
    print ("Constructed model with {0} modes of variation".format(t))
    
  
    return model






  # for i in range(30):
    #     x= eigenVectors[5,:]
       
    #     xxx=eigenvalue[3]
       
       
    #     b = np.dot(x,xxx)
    #     y = mean_shape + b

    #     plt.axis([-216, 304, -216, 304])
    #     plt.plot(y[0:29],y[30:59],".")
    #     plt.show()
   
    # return (eigenvalue[:t], eigenVectors[:,:t], mean_shape, t)


# x= eigenVectors[5,:]
   
# xxx=eigenvalue[3]
   
   
# b = np.dot(x,xxx)
# y = mean_shape + b

        
'''
print(LandMarkFinalMatrix.shape[0])

mean_shapex= np.mean(LandMarkFinalMatrix[0],axis=1)
mean_shapey= np.mean(LandMarkFinalMatrix[1],axis=1)

mean_shape=np.array([mean_shapex,mean_shapey]).T

gs=LandMarkFinalMatrix[0,:,2]
oo=mean_shape[:,0]

#subs=np.subtract(LandMarkFinalMatrix[0,:,2],mean_shape[:,0])


#subtract = np.subtract(LandMarkFinalMatrix[0],mean_shape[0])

sub22= []
for i in range (LandMarkFinalMatrix.shape[2]):
    sub22.append(np.subtract(LandMarkFinalMatrix[0,:,i],mean_shape[:,0]))


hh=np.array(sub22).T

dots=np.dot(hh,hh.T)/LandMarkFinalMatrix.shape[2]

    
    
sub23= []
for i in range (LandMarkFinalMatrix.shape[2]):
    sub23.append(np.subtract(LandMarkFinalMatrix[1,:,i],mean_shape[:,1]))
    

hh2=np.array(sub23).T

dots2=np.dot(hh2,hh2.T)/LandMarkFinalMatrix.shape[2]

#var=np.concatenate(dots,dots2)
covar= np.dstack([dots,dots2]).T

D=np.linalg.eig(covar)

eigenvector= np.array(D[1])   
eigenvalue= np.array(D[0]) 



for i in range(eigenvector.shape[1]):
    x= eigenvector[0,:,i]
    y=eigenvector[1,:,i]
    xxx=eigenvalue[0,i]
    yyy=eigenvalue[1,i]
        
    b = np.dot(x,xxx)
    c =np.dot(y,yyy)
    shapeexample = (mean_shape+ np.vstack((b,c)).T).T
        
    
    plt.plot(shapeexample[0,:],shapeexample[1,:])
    plt.show()
'''    

    
    



