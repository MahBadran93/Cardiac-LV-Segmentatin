'''
Build a grey-level profile length 2k+1 along the normal to a point.
'''
import math
import numpy as np


class Profiler(object):
    '''
    Class that creates a normal to input points and computes the 2k nearest
    pixels to that normal.
    '''
    def __init__(self, k=0):
        self.reset(k)

    def reset(self, k):
        '''
        Reset profiler variables
        '''
        self.k = k
        self.normal = None

    def sample(self, points):
        '''
        Get a sample from self.k points on each side of the normal between
        the triple in points.

        in: triple previous_point, point, and next_point
        out: list of tuples along the normal through point
        '''
        # compute the normal
        self.normal = self._compute_normal(points)
        
        # sample along the normal
        # print ("normal " , self.normal)
        return self._sample(points[1])

    def profile(self, image, points):
        '''
        Compute the distance to normal for each pixel in frame. Return the
        greyscale intensity of the 2k+1 nearest pixels to normal.

        out: list of grey-levels
        '''
        # print (points )

        # print ("prooooofile", image.shape, image[0.15632709.astype(int) ,-0.98770534.astype(int)])
    
        
        greys = np.asarray([float(image[r.astype(np.int), c.astype(np.int)]) for r, c in self.sample(points)])
        return self._normalize(self._derive(greys))

    def _sample(self, starting_point):
        '''
        Returns 2k+1 points along the normal
        '''
        positives = []
        negatives = []
        start = [(int(starting_point[0]), int(starting_point[1]))]

        i = 1
        while len(positives) < self.k:
            new = (starting_point[0] - i*self.normal[0], starting_point[1] - i*self.normal[1])
            if (new not in positives) and (new not in start):
                positives.append(new)
            i += 1

        i = 1
        while len(negatives) < self.k:
            new = (starting_point[0] + i*self.normal[0], starting_point[1] + i*self.normal[1])
            if (new not in negatives) and (new not in start):
                negatives.append(new)
            i += 1

        negatives.reverse()

        return np.array(negatives + start + positives)

    def _compute_normal(self, points):
        '''
        Compute the normal between three points.
        '''
        prev, curr, nex = points
        return self._normal(prev, nex)

    def _normal(self, a, b):
        '''
        Compute the normal between two points a and b.

        in: tuple coordinates a and b
        out: 1x2 array normal
        '''
        d = b - a
        tx, ty = d/math.sqrt(np.sum(np.power(d, 2)))
        return np.array([-1*ty, tx])

    def _derive(self, profile):
        '''
        Get derivative profile by computing the discrete difference.
        See Hamarneh p.13.
        '''
        return np.diff(profile)

    def _normalize(self, vector):
        '''
        Normalize a vector such that its sum is equal to 1.
        '''
        div = np.sum(np.absolute(vector)) if np.sum(np.absolute(vector)) > 0 else 1
        return vector/div
