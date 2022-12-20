import skimage.io as io
import skimage as sk
from skimage.color import rgb2gray
from skimage.transform import resize
import time
import numpy as np
import os


from unitTesting import UnitTest
import adaboost.adaboost as ab


img = io.imread('./test.jpg')
img = rgb2gray(img)
ut = UnitTest(img)

positiveDataset = np.load("./olivetti_faces.npy")
personID = np.load("./olivetti_faces_target.npy")


newPositiveDataset = np.empty(shape=(positiveDataset.shape[0], 24, 24))
#resize images to 24x24
for i in range(len(positiveDataset)):
    newImg = np.array(positiveDataset[i])
    
    finalImg = resize(newImg, (24, 24))
    newPositiveDataset[i] = finalImg

positiveDataset = newPositiveDataset

# for i in range(10):
#     io.imshow(positiveDataset[i])
#     io.show()

# print(positiveDataset.shape)

directory = "./dataset/NegativeSet"
negativeDataset = np.empty(shape=(len(os.listdir(directory)), 24, 24))
i = 0
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        curImg = io.imread(os.path.join(directory, filename), as_gray=True)
        newImg = resize(curImg, (24, 24))
        negativeDataset[i] = newImg
        i += 1
# negativeDataset = negativeDataset.reshape(len(os.listdir(directory)), 64, 64)
# print(negativeDataset.shape)


# for i in range(10):
#     io.imshow(negativeDataset[i])
#     io.show()




posDataset, negDataset = ab.preprocessImages(positiveDataset, negativeDataset)
print(posDataset.shape)
print(negDataset.shape)

#Now we have the positive and negative datasets with size 24x24
print("Starting to train the classifier")
start = time.time()
classifier = ab.AdaBoost()