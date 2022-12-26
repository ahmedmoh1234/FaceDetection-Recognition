import numpy as np
from typing import List,Tuple
from functools import partial
from skimage.color import rgb2gray
from multiprocessing import Pool

from adaboost.haarlikefeatures import HaarLikeFeature
#================================================================================================
#================================================================================================
def determineFeatures(img, threshold, minFeatureWidth, minFeatuerHeight, maxFeatureWidth, maxFeatureHeight) -> List[HaarLikeFeature]:
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
        for haarType in HaarLikeFeature.HaarType:
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
                            haarFeature1 = HaarLikeFeature(x,y,width,height,haarType,threshold, 1)
                            haarFeature2 = HaarLikeFeature(x,y,width,height,haarType,threshold,-1)

                            haarFeatures.append(haarFeature1)
                            haarFeatures.append(haarFeature2)
                            # count += 1
            # print(f"Feature type = {haarType.name}, count = {count}")
        print(f"Total number of features = {len(haarFeatures)}")
        return haarFeatures

def computeError(y,yPred,w):
    # y : actual value
    # yPred : predicted value
    # w : weights of samples
    # error = sum of weight of misclassified samples / sum of weight of all samples
    return np.sum(w[y!=yPred])/np.sum(w)

def computeAlpha(error):
    # compute alpha (weight) for the current weak classifier m
    return 0.5*np.log((1-error)/error)

def updateWeights(y,yPred,w,alpha):
    # update weights of samples
    # w = w*exp(-alpha*y*yPred)
    w = w * np.exp(-alpha * (np.not_equal(y,yPred)).astype(int))
    return w



def integralImage(img):
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

def preprocessImages(positiveImgs,negativeImgs):
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

    positiveImgs = positiveImgs * 255
    negativeImgs = negativeImgs * 255

    #zero mean and unit variance
    # positiveImgs = (positiveImgs - np.mean(positiveImgs))/np.std(positiveImgs)
    # negativeImgs = (negativeImgs - np.mean(negativeImgs))/np.std(negativeImgs)

    #remove images with variance less than 1
    elementsToRemove = []
    i = 0
    for img in positiveImgs:
        if np.var(img)<1:
            elementsToRemove.append(i)
        i+=1

    newPositiveImgs = np.delete(positiveImgs,elementsToRemove,axis=0)

    elementsToRemove = []
    i = 0
    for img in negativeImgs:
        if np.var(img)<1:
            elementsToRemove.append(i)
        i+=1

    newNegativeImgs = np.delete(negativeImgs,elementsToRemove,axis=0)

    return newPositiveImgs,newNegativeImgs

#================================================================================================
#================================================================================================

class AdaBoost():

    #weakClassifier : weak classifier (Gm)
    #M : number of boosting iterations

    def __init__(self):
        self.alphas = []
        self.weakClassifiers = []
        self.M = None
        self.trainingErrors = []
        self.predictionErrors = []

    #learn tak
    def learn(self,positiveImgs,negativeImgs,threshold,minFeatureWidth, minFeatureHeight,maxFeatureWidth,maxFeatureHeight, nClassifiers):
        # positiveImgs : list of positive images
        # negativeImgs : list of negative images
        # threshold : threshold value
        # maxFeatureWidth : maximum width of a feature
        # maxFeatureHeight : maximum height of a feature
        
        #calculate integral images of positive and negative images
        print("Calculating integral images...")
        positiveIntegralImages = [integralImage(img) for img in positiveImgs]
        negativeIntegralImages = [integralImage(img) for img in negativeImgs]
        print("Done!\n")

        # print(f"shape of positive integral image: {positiveIntegralImages[0].shape}")
        
        #calculate number of positive and negative samples
        nPositive = len(positiveImgs)
        nNegative = len(negativeImgs)
        
        #calculate number of features
        nImages = nPositive + nNegative
        
        #calculate number of rows and columns of the images
        height,width = positiveImgs[0].shape
        
        #============CHANGED================
        # maxFeatureWidth = min(maxFeatureWidth,width)
        # maxFeatureHeight = min(maxFeatureHeight,height)
        
        
        #create inital weights of samples and labels
        print("Creating initial weights and labels...")
        weightPositive = np.ones(nPositive) * 1. /(2 * nPositive)
        weightNegative = np.ones(nNegative) * 1. /(2 * nNegative)
        
        #============CHANGED================
        weights = np.hstack((weightPositive,weightNegative))
        
        #============CHANGED================
        labels = np.hstack((np.ones(nPositive),np.ones(nNegative) * -1))
        print("Done!\n")
        
        integralImages = positiveIntegralImages + negativeIntegralImages

        
        #calculate all possible haar features
        print("Calculating all possible haar features...")
        haarFeatures = determineFeatures(positiveImgs[0],threshold,minFeatureWidth,minFeatureHeight,maxFeatureWidth,maxFeatureHeight)
        print("Done!\n")
        
        #calculate number of haar features
        nHaarFeatures = len(haarFeatures)

        featureIndices = list(range(nHaarFeatures))

        
        #create feature matrix
        votes = np.zeros((nImages,nHaarFeatures))
        
        #calculate votes of all haar features for all samples
        print("Calculating votes of all haar features for all samples...")


        pool = Pool(processes=None)
        for i in range(nImages):
            votes[i,:] = np.array(list(pool.map(partial(getFeatureVote, integralImage=integralImages[i]), haarFeatures)))


        # votes1 = np.zeros((nImages,nHaarFeatures))
        # for i in range(nImages):
        #     votes1[i,:] = np.array([haarFeatures[j].getVote(integralImages[i]) for j in range(nHaarFeatures)])

        # if votes1.all() == votes.all():
        #     print("Multiprocessing works!")
        # else:
        #     print("Multiprocessing doesn't work!")
        print("Done!\n")
                
        #select classifiers

        classifiers = []
        print(f"Training {nHaarFeatures} classifiers...")
        for i in range(nClassifiers):
            
            print("\r", f"Training classifier {i+1} of {nClassifiers}...", end="")

            classificationError = np.zeros(len(featureIndices))
            
            #normALIZE WEIGHTS
            weights = weights/np.sum(weights)
            
            #select classifier with minimum weighted error  
            for j in range(len(featureIndices)):
                fIndex = featureIndices[j]
                #calculate weighted error
                classificationError[j] = np.sum(weights[labels!=votes[:,fIndex]])
                
                
            minErrorIndex = np.argmin(classificationError)
            bestError = classificationError[minErrorIndex]
            bestFeatureIndex = featureIndices[minErrorIndex]
            
            #set feature weight
            bestFeature = haarFeatures[bestFeatureIndex]
            if bestError == 0:
                bestError = 0.1
            bestFeature.weight = 0.5 * np.log((1-bestError)/bestError)
            
            classifiers.append(bestFeature)
            # update image weights
            weights[labels!=votes[:,bestFeatureIndex]] *= np.exp(bestFeature.weight)
            weights[labels==votes[:,bestFeatureIndex]] *= np.exp(-bestFeature.weight)
            
            #normalize weights
            weights = weights/np.sum(weights)
            
            #remove selected feature from feature list
            featureIndices.remove(bestFeatureIndex)

        return classifiers

def getFeatureVote(feature : HaarLikeFeature ,integralImage):
    # print(f"getfeature vote int img shape: {integralImage.shape}")
    return feature.getVote(integralImage)


def getVotes(haarFeatures ,integralImage):
    #Threshold = 0 because each feature votes for 1 or -1
    return 1 if np.sum(c.getVote(integralImage) for c in haarFeatures) >= 0 else 0 # type: ignore
#================================================================================================