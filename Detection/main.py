import skimage.io as io
from skimage.color import rgb2gray
import time
import numpy as np

from unitTesting import UnitTest

img = io.imread('./test.jpg')
img = rgb2gray(img)


#comment the following line to skip unit testing
ut = UnitTest(img)

