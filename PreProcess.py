import SimpleITK as sitk
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import os
from multiprocessing import pool
import numpy as np
from skimage.transform import resize
import cv2
import scipy
import skimage as sk
from PIL import Image


referenceImg = sitk.ReadImage('../training/patient001/patient001_frame01_gt.nii.gz')




def SampleTest1(img):
   
    dimension = referenceImg[:,:,1].GetDimension()
    #new_size = [100, 100]
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()
    reference_size = [128]*dimension 
    reference_physical_size = np.zeros(dimension)
    reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]
    reference_image = sitk.Image(reference_size, img.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetDirection(reference_direction)
    reference_image.SetSpacing(reference_spacing)
    #result = sitk.GetArrayFromImage(sitk.Resample(img, reference_image))  
    result = sitk.Resample(img, reference_image)
    return result

def crop_height(new_height, height):
    remove_y_top = (height - new_height) // 2
    remove_y_bottom = height - new_height - remove_y_top
    return remove_y_top, remove_y_bottom


def crop_width(new_width, width):
    remove_x_left = (width - new_width) // 2
    remove_x_right = width - new_width - remove_x_left
    return remove_x_left, remove_x_right




def SampleTest2(img):
    
    
    image = sitk.GetArrayFromImage(img)    
    img_res = sk.transform.resize(image,(128,128))
    image_rescaled = sk.transform.rescale(img_res, 1, anti_aliasing=False)
   
    #imageResc = Image.fromarray(image_rescaled)
    #topBott = crop_height(image_rescaled.shape[0],image.shape[0])
    #leftright = crop_width(image_rescaled.shape[1],image.shape[1])
    #imageResc.crop((leftright[0],topBott[0],leftright[1],topBott[1]))
    image_rescaled[50:80,50:80]
   
    return image_rescaled




















"""

def resize_image(image, old_spacing, new_spacing, order=3):
    new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                 int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                 int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
    return resize(image, new_shape, order=order, mode='edge')

def convert_to_one_hot(seg):
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res


def preprocess_image(itk_image, is_seg=False, spacing_target=(1, 0.5, 0.5), keep_z_spacing=False):
    spacing = np.array(itk_image.GetSpacing())[[2, 1, 0]]
    image = sitk.GetArrayFromImage(itk_image).astype(float)
    if keep_z_spacing:
        spacing_target = list(spacing_target)
        spacing_target[0] = spacing[0]
    if not is_seg:
        order_img = 3
        if not keep_z_spacing:
            order_img = 1
        image = resize_image(image, spacing, spacing_target, order=order_img).astype(np.float32)
        image -= image.mean()
        image /= image.std()
    else:
        tmp = convert_to_one_hot(image)
        vals = np.unique(image)
        results = []
        for i in range(len(tmp)):
            results.append(resize_image(tmp[i].astype(float), spacing, spacing_target, 1)[None])
        image = vals[np.vstack(results).argmax(0)]
    return image

#plt.imshow(gg[60,:,:])
#plt.show()
"""