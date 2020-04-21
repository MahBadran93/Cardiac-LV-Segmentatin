from AutomateLandMarks import  LandMarks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def Build_Model():
    
    
    #object to construct the shape list (1881) images 
    showLand = LandMarks() 
    # return a list of sampled slices (30,1881)
    landMarkedShapes,originalShape = showLand.GenerateSampleShapeList()
    # convert the list to nunmpy array  (2,30,1881) 
    #which 30 number of point in each shape and 2 is the coordinate
    landMarkedShapesR = np.stack(landMarkedShapes,axis=0 ) 
    LandMarkFinalMatrix =landMarkedShapesR.T
    
    '''
    the	landmark matrices are converted	to landmark	vectors.This means that	
	30	2-dimensional landmarks	(30,2), a (60,1) column vector	will be obtained. 
    (2,30,1881) => (1881,60)

    '''
    Landmarkcoulmn= np.vstack((LandMarkFinalMatrix[0,:,:],LandMarkFinalMatrix[1,:,:])).T
    
    
    pca = PCA()
    reduced = pca.fit(Landmarkcoulmn)
    eigenVectors =reduced.components_
    eigenvalue=reduced.explained_variance_
    mean_shape=reduced.mean_
    
    
    # Find number of modes(eigenvalue) required to describe the most important variance of the data 
    t = 0 # # store the number of requierd eigen value
    for i in range(len(eigenvalue)):
      if sum(eigenvalue[:i]) / sum(eigenvalue) < 0.99:
        t = t + 1
      else: break
  
    

    for i in range(30):
        x= eigenVectors[i,:]
    
        xxx=eigenvalue[i]
        
    
        b = np.dot(x,xxx)
        
        shapeexample = mean_shape + b
            
     #  plt.axis([-216, 304, -216, 304])
        plt.plot(shapeexample[0:29],shapeexample[30:59],".")
        plt.show()
        
        
        
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

    
    



