from AutomateLandMarks import  LandMarks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
 

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
    mean_shape=pca_model.mean_
    
    
    # Find number of modes(eigenvalue) required to describe the most important variance of the data 
    t = 0
    for i in range(len(eigenvalue)):
      if sum(eigenvalue[:i]) / sum(eigenvalue) < 0.99:
          t = t + 1
      else: break
  
    print ("Constructed model with {0} modes of variation".format(t))
    
  
    return (t,pca_model,Landmarkcoulmn)






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

    
    



