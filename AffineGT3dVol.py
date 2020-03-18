import matplotlib.pyplot as plt
import numpy as np 
import loadnif as nf

from scipy.ndimage.interpolation import affine_transform

imgESGT = nf.loadNifti('../training/patient002/patient002_frame01_gt.nii.gz')

nimages = imgESGT.shape[2]
img_height, img_width = 256, 256
bg_val = -1 # Some flag value indicating the background.

# Random test images.


stacked_height = 2*img_height
stacked_width  = img_width + int((nimages-1)*img_width/2)
stacked = np.full((stacked_height, stacked_width), bg_val)

# Affine transform matrix.
T = np.array([[1,-1],
              [0, 1]])

for i in range(nimages):
    # The first image will be right most and on the "bottom" of the stack.
    o = (nimages-i-1) * img_width/2
    out = affine_transform(imgESGT[:,:,i], T, offset=[o,-o],
                           output_shape=stacked.shape, cval=bg_val)
    stacked[out != bg_val] = out[out != bg_val]

plt.imshow(stacked, cmap=plt.cm.viridis)
plt.show()