from AutomateLandMarks import  LandMarks
import saveDataSet as saveData
import loadnif as nif
import numpy as np
import matplotlib.pyplot as plt
import cv2
import  shapely as shp
import shapely.geometry as gmt
import pylab as pl

#.............................Testing.....................................................

img4D = nif.loadNifti('../training/patient001/patient001_4d.nii.gz')
imgES= nif.loadNifti('../training/patient001/patient001_frame12.nii.gz')
imgED =nif.loadNifti('../training/patient001/patient001_frame01.nii.gz') 
imgESGT = nif.loadNifti('../training/patient001/patient001_frame01_gt.nii.gz')
imgESGT1 = nif.loadNifti('../training/patient002/patient002_frame01_gt.nii.gz')

#........... Create Training Set...........................
#print(len(landMarks.getLandMarksCoords()))
#landMarks.getLandMarksCoords()
#plt.plot(np.array(landMarks.getLandMarksCoords()[2])[:,0], np.array(landMarks.getLandMarksCoords()[2])[:,1], '.')
#plt.axis([-216, 304, -216, 304])
#plt.show()
#.....................Save png of images .........................
#saveData.saveDataSet()

fff = np.array([[1,2],[3,4],[5,6]])
ttt = np.add(fff,[1,1])
print(ttt)

showLand = LandMarks()
shapeList,shapeCentroids = showLand.getLandMarksCoords()


#coords = [p[:][0] for p in shapeList[500]]

#x = np.array(shapeList[500])[:,0]
#y = np.array(shapeList[500])[:,1]
"""
plt.figure()
x1 = [p[0] for p in shapeList[120]]
y1 = [p[1] for p in shapeList[120]]
plt.plot(x1,y1,'.')
poly = gmt.Polygon([p[0],p[1]] for p in shapeList[120])
x,y = poly.convex_hull.exterior.coords.xy
#t,y = poly.convex_hull.coords.xy
#t,n = poly.contour.exterior.coords.xy
plt.plot(x,y)    
plt.show()
#pl.figure(figsize=(10,10))
#_ = pl.plot(poly,'o', color='#f16824')
"""
def findCentroid(x,y):
    length = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    return [np.round(sum_x/length), np.round(sum_y/length)] # returm coords of the centroid for each shape 

for i in range(len( shapeList)):
    if(len(shapeList[i]) !=0):
        #plt.figure()
        x1 = [p[0] for p in shapeList[i]]
        y1 = [p[1] for p in shapeList[i]]
        #print('shapeCentroid',findCentroid(x1,y1))
        #plt.plot(x1,y1,'.')
        poly = gmt.Polygon([p[0],p[1]] for p in shapeList[i])
        x,y = poly.convex_hull.exterior.coords.xy
        print('length',len(x),'length2',len(x1))
        #plt.axis([-216, 304, -216, 304])
        #plt.plot(x,y)
        #t,y = poly.convex_hull.coords.xy
        #t,n = poly.contour.exterior.coords.xy
        #plt.show()
    else:
        print('empty shape')

    
#polygon = gmt.Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
#points = np.array(polygon)





"""
listDim = []
count = 0

swap = len(shapeList[3])
for i in range(len(shapeList)):
    listDim.append(len(shapeList[i]))
    if(len(shapeList[i])!=0 and len(shapeList[i])!=4 and len(shapeList[i])!=7 and len(shapeList[i])!=9):
        if(len(shapeList[i]) < swap):
            swap = len(shapeList[i])
            count = i

print(count, shapeList[count])
"""         
#np.savetxt('content/nCentroids.txt',shapeList , fmt='%s')
#np.savetxt('content/listDim.txt',listDim , fmt='%s')
#np.savetxt('content/nCentroids.txt',shapeCentroids , fmt='%s')


#print(LoadAllGT().shape)        
 

#################################################################
"""
coords = cv2.imread('plotXYshape.png')
fig = plt.figure
fig,((x1,x2),(x3,x4)) = plt.subplots(2,2)
x1.set_title('Ground Truth')
x1.imshow(imgESGT[:,:,2])
x2.set_title('detect contours  from GT')  
x2.imshow(cv2.Canny(np.uint8(imgESGT[:,:,2]),0,1))
imgESGT[:,:,2][imgESGT[:,:,2] < 3] = 0
x3.set_title('detect LV contour only from GT')   
x3.imshow(cv2.Canny(np.uint8(imgESGT[:,:,2]),0,1))
x4.set_title('plotted x,y coords extracted from shape')
x4.imshow(coords)
"""

#plt.show()

#for i in range(imgGT.shape[2]):
    #displaySlices(imgGT,i) 
#displaySegmentedGTSlices(imgRe,5)
