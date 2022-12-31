import numpy as np
import numpy.typing as npt
from typing import List,Tuple
from skimage.color import rgb2gray

# from adaboost.haarlikefeatures import HaarLikeFeature

import adaboost.haarlikefeatures as HLF

def determineFeatures(img, threshold, minFeatureWidth, minFeatuerHeight, maxFeatureWidth, maxFeatureHeight) -> List[HLF.HaarLikeFeature]:
        # img : input image
        # n : number of rows
        # m : number of columns
        # haarFeatures : list of haar like features
        # haarFeature : haar like feature
        # haarType : type of haar like feature
        # threshold : threshold value
        # polarity : polarity of the feature (1 or -1)
        n,m = img.shape
        haarFeatures = []
        for haarType in HLF.HaarLikeFeature.HaarType:
            # count  = 0
            featureWidthStart = max(minFeatureWidth,haarType.value[0])
            for width in range(featureWidthStart,maxFeatureWidth,haarType.value[0]):
                featureHeightStart = max(minFeatuerHeight,haarType.value[1])
                for height in range(featureHeightStart,maxFeatureHeight ,haarType.value[1]):
                    #============CHANGED================
                    for x in range(m - width):
                        for y in range(n - height):
                            if x + width > m or y + height > n:
                                print('==========Error')
                                print(f"x = {x}, y = {y}, width = {width}, height = {height} Image size = {img.shape}")
                            haarFeature1 = HLF.HaarLikeFeature(x,y,width,height,haarType,threshold, 1)
                            haarFeature2 = HLF.HaarLikeFeature(x,y,width,height,haarType,threshold,-1)

                            haarFeatures.append(haarFeature1)
                            haarFeatures.append(haarFeature2)
                            # count += 1
            # print(f"Feature type = {haarType.name}, count = {count}")
        print(f"Total number of features = {len(haarFeatures)}")
        return haarFeatures

def integralImage(img) -> np.ndarray:
    # img : image
    # integralImage : integral image of the input image
    # integralImage[i,j] = sum of all pixels in the rectangle from (0,0) to (i,j)
    # test time complexity of this function


    #np.cumsum() is much faster than the for loop
    # intImg = np.zeros(img.shape)
    # for i in range(0,img.shape[0]):
    #     for j in range(0,img.shape[1]):
    #         if i==0 and j==0:
    #             intImg[i,j] = img[i,j]
    #         elif i==0 and j!=0:
    #             intImg[i,j] = img[i,j] + intImg[i,j-1]
    #         elif i!=0 and j==0:
    #             intImg[i,j] = img[i,j] + intImg[i-1,j]
    #         else:
    #             intImg[i,j] = img[i,j] + intImg[i-1,j] + intImg[i,j-1] - intImg[i-1,j-1]
    # return intImg

    # intImg2 = np.zeros(img.shape)
    # for i in range(1,img.shape[0]):
    #     for j in range(1,img.shape[1]):
    #             intImg2[i,j] = img[i,j] + intImg2[i-1,j] + intImg2[i,j-1] - intImg2[i-1,j-1]

    # return np.cumsum(np.cumsum(img,axis=0),axis=1)


    #calculate integral image with first row and first column as zeros
    #because this is a convention
    ii = np.zeros((img.shape[0]+1,img.shape[1]+1))
    # print(ii.shape)
    ii[1:,1:] = np.cumsum(np.cumsum(img,axis=0),axis=1)
    return ii
    # return np.cumsum(np.cumsum(np.pad(img,((1,0),(1,0)),'constant'),axis=0),axis=1) # type: ignore

def preprocessImages(positiveImgs,negativeImgs) -> Tuple[np.ndarray,np.ndarray]:
    # positiveImgs : list of positive images
    # negativeImgs : list of negative images

    #make all images grayscale using numpy
    for i in range(len(positiveImgs)):
        if len(positiveImgs[i].shape)!=2:
            positiveImgs[i] = rgb2gray(positiveImgs[i])

    for i in range(len(negativeImgs)):
        if len(negativeImgs[i].shape)!=2:
            negativeImgs[i] = rgb2gray(negativeImgs[i])


    #normalize all images
    positiveImgs = positiveImgs/np.max(positiveImgs)
    negativeImgs = negativeImgs/np.max(negativeImgs)

    # positiveImgs = positiveImgs * 255
    # negativeImgs = negativeImgs * 255

    #zero mean and unit variance
    positiveImgs = (positiveImgs )/np.std(positiveImgs)
    negativeImgs = (negativeImgs )/np.std(negativeImgs)

    #remove images with variance less than 1
    # elementsToRemove = []
    # i = 0
    # for img in positiveImgs:
    #     if np.var(img)<1:
    #         elementsToRemove.append(i)
    #     i+=1

    # newPositiveImgs = np.delete(positiveImgs,elementsToRemove,axis=0)

    # elementsToRemove = []
    # i = 0
    # for img in negativeImgs:
    #     if np.var(img)<1:
    #         elementsToRemove.append(i)
    #     i+=1

    # newNegativeImgs = np.delete(negativeImgs,elementsToRemove,axis=0)

    newPositiveImgs = positiveImgs
    newNegativeImgs = negativeImgs

    return newPositiveImgs,newNegativeImgs

def getFeatureVote(feature : HLF.HaarLikeFeature ,integralImage) -> float:
    # print(f"getfeature vote int img shape: {integralImage.shape}")
    return feature.getVote(integralImage)


def getVotes(haarFeatures ,integralImage) -> int:
    #Threshold = 0 because each feature votes for 1 or -1
    votes = 0
    for c in haarFeatures:
        votes += c.getVote(integralImage)
        if votes < 0:
            return 0
    return 1
    # return 1 if np.sum(c.getVote(integralImage) for c in haarFeatures) >= 0 else 0 # type: ignore

