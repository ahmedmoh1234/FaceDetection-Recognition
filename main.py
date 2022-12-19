import skimage.io as io
from adaboost.adaboost import integralImage
import time
import numpy as np
from skimage.color import rgb2gray

from unitTesting import UnitTest

img = io.imread('./test.jpg')
img = rgb2gray(img)


ut = UnitTest(img)

