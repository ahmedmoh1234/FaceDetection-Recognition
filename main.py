import skimage.io as io
from adaboost.adaboost import integralImage
import time
import numpy as np
from skimage.color import rgb2gray

from unitTesting import UnitTest
from adaboost.haarlikefeatures import HaarLikeFeature

img = io.imread('./test.jpg')
img = rgb2gray(img)


#comment the following line to skip unit testing
ut = UnitTest(img)

