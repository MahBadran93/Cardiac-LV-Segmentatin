
import argparse
import numpy as np
import glob
import re
#from log import print_to_file
from scipy.fftpack import fftn, ifftn
from skimage.feature import peak_local_max, canny
from skimage.transform import hough_circle
#import cPickle as pickle
#from paths import TRAIN_DATA_PATH, LOGS_PATH, PKL_TRAIN_DATA_PATH, PKL_TEST_DATA_PATH
#from paths import TEST_DATA_PATH

def extract_roi(data, pixel_spacing, minradius_mm=15, maxradius_mm=65, kernel_width=5, center_margin=8, num_peaks=10,
                    num_circles=10, radstep=2):
    
    """
    Returns center and radii of ROI region in (i,j) format
    """
    # radius of the smallest and largest circles in mm estimated from the train set
    # convert to pixel counts
    minradius = int(minradius_mm / pixel_spacing)
    maxradius = int(maxradius_mm / pixel_spacing)

    ximagesize = data.shape[0]
    yimagesize = data.shape[1]

    xsurface = np.tile(range(ximagesize), (yimagesize, 1)).T
    ysurface = np.tile(range(yimagesize), (ximagesize, 1))
    lsurface = np.zeros((ximagesize, yimagesize))

    allcenters = []
    allaccums = []
    allradii = []

    for i in range(data.shape[2]):
        ff1 = fftn(data[:,:,i])
        fh = np.absolute(ifftn(ff1[:, :]))
        fh[fh < 0.1 * np.max(fh)] = 0.0
        image = 1. * fh / np.max(fh)


        # find hough circles and detect two radii
        edges = canny(image, sigma=3)
        hough_radii = np.arange(minradius, maxradius, radstep)
        hough_res = hough_circle(edges, hough_radii)

        if hough_res.any():
            centers = []
            accums = []
            radii = []

            for radius, h in zip(hough_radii, hough_res):
                # For each radius, extract num_peaks circles
                peaks = peak_local_max(h, num_peaks=num_peaks)
                centers.extend(peaks)
                accums.extend(h[peaks[:, 0], peaks[:, 1]])
                radii.extend([radius] * num_peaks)

            # Keep the most prominent num_circles circles
            sorted_circles_idxs = np.argsort(accums)[::-1][:num_circles]

            for idx in sorted_circles_idxs:
                center_x, center_y = centers[idx]
                allcenters.append(centers[idx])
                allradii.append(radii[idx])
                allaccums.append(accums[idx])
                brightness = accums[idx]
                lsurface = lsurface + brightness * np.exp(
                    -((xsurface - center_x) ** 2 + (ysurface - center_y) ** 2) / kernel_width ** 2)

    lsurface = lsurface / lsurface.max()

    # select most likely ROI center
    roi_center = np.unravel_index(lsurface.argmax(), lsurface.shape)

    # determine ROI radius
    roi_x_radius = 0
    roi_y_radius = 0
    for idx in range(len(allcenters)):
        xshift = np.abs(allcenters[idx][0] - roi_center[0])
        yshift = np.abs(allcenters[idx][1] - roi_center[1])
        if (xshift <= center_margin) & (yshift <= center_margin):
            roi_x_radius = np.max((roi_x_radius, allradii[idx] + xshift))
            roi_y_radius = np.max((roi_y_radius, allradii[idx] + yshift))

    if roi_x_radius > 0 and roi_y_radius > 0:
        roi_radii = roi_x_radius, roi_y_radius
    else:
        roi_radii = None

    return roi_center