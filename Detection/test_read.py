import skimage.io as io
import numpy as np

import os
_file_ = os.path.abspath(__file__)
dirname = os.path.dirname(_file_)
filename = os.path.join(dirname, 'test.jpg')

img = io.imread(filename)
print(img.shape)
