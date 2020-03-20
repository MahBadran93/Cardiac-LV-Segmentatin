import numpy as np 
from skimage.color import rgb2gray
import cv2 
import matplotlib.pyplot as plt 
from scipy import ndimage
from sklearn.cluster import KMeans
import mpmath as math 
from sympy.geometry import *
import loadnif as nf
from skimage.color import rgb2gray
import imageio as save 



imgESGT = nf.loadNifti('../training/patient003/patient003_frame15_gt.nii.gz')

for i in range(imgESGT.shape[2]):
    slice1Copy = np.uint8(imgESGT[:,:,i])
    #gray = cv2.medianBlur(slice1Copy, 5)
    edgedImg = cv2.Canny(slice1Copy,0,1)
    #ret, th_img = cv2.threshold(edgedImg, thresh=1, maxval=255, type=cv2.THRESH_BINARY)

    circles = cv2.HoughCircles(edgedImg,cv2.HOUGH_GRADIENT,1,int(imgESGT[:,:,i].shape[0]/6),param1=100,param2=30,minRadius=10,maxRadius=40)
    circles	= np.uint16(np.around(circles))
    for x,y,r in circles[0, :]:
        print('x, y, r:', x, y, r)
        border = 2

        cv2.circle(edgedImg, (x, y), r, (255, 0, 255), border)
        cv2.circle(edgedImg, (x, y), 2, (0, 0, 255), 3)

        height, width = edgedImg.shape
        print('height, width:', height, width)

                    # calculate region to crop
        x1 = max(x-r - border//2, 0)      # eventually  -(border//2+1)
        x2 = min(x+r + border//2, width)  # eventually  +(border//2+1)
        y1 = max(y-r - border//2, 0)      # eventually  -(border//2+1)
        y2 = min(y+r + border//2, height) # eventually  +(border//2+1)
        print('x1, x2:', x1, x2)
        print('y1, y2:', y1, y2)

        # crop image 
        image = edgedImg[y1:y2,x1:x2]
        print('height, width:', image.shape)

    #test_circ = np.int16(np.around(circles))
    #s = edgedImg[test_circ[0,0,1]-test_circ[0,0,2]:test_circ[0,0,1]+test_circ[0,0,2], test_circ[0,0,0]-test_circ[0,0,2]:test_circ[0,0,0]+test_circ[0,0,2]]

    """
    for	n in circles[0,:]:
	    t = cv2.circle(edgedImg,(n[0],n[1]),n[2],(255,255,0),2)
    """
    cv2.imshow("HoughCirlces",	edgedImg)
    cv2.waitKey(1000)
        
    

"""
testImage = cv2.imread('1.jpeg')
edgeImg = cv2.imread('index.png')
ss = np.array(testImage)
cv2.circle(testImage,(int(ss.shape[0]/2),int(ss.shape[1]/2)),15,1,thickness=1)
events = [i for i in dir(cv2) if 'EVENT' in i ]
print(events)

gryImg = rgb2gray(testImage)
gryedgeImg = rgb2gray(edgeImg)

gray_r = gryImg.reshape(gryImg.shape[0]*gryImg.shape[1])

for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 3
    elif gray_r[i] > 0.5:
        gray_r[i] = 2
    elif gray_r[i] > 0.25:
        gray_r[i] = 1
    else:
        gray_r[i] = 0            

gray = gray_r.reshape(gryImg.shape[0], gryImg.shape[1])  

###################### Edge detection 
sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
print(sobel_horizontal, 'is a kernel for detecting horizontal edges')
 
sobel_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])
print(sobel_vertical, 'is a kernel for detecting vertical edges')

convHor = ndimage.convolve(gryedgeImg , sobel_vertical , mode="reflect")

####################### end

############################### clustering 
pic = plt.imread('1.jpeg')/255  # dividing by 255 to bring the pixel values between 0 and 1
pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
kmeans = KMeans(n_clusters=8, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]

cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
plt.imshow(cluster_pic)



#print(clusterdPic.shape)
#plt.show()
"""