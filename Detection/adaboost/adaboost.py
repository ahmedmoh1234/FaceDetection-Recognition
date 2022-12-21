import numpy as np
from typing import List,Tuple
from skimage.color import rgb2gray
from multiprocessing import Pool

from adaboost.haarlikefeatures import HaarLikeFeature
#================================================================================================
#================================================================================================
def determineFeatures(img, threshold, maxFeatureWidth, maxFeatureHeight) -> List[HaarLikeFeature]:
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
            featureWidthStart = haarType.value[0] 
            for width in range(featureWidthStart,maxFeatureWidth+1 ,haarType.value[0]):
                featureHeightStart = haarType.value[1]
                for height in range(featureHeightStart,maxFeatureHeight+1 ,haarType.value[1]):
                    for x in range(m - width+1):
                        for y in range(n - height+1):
                            if x + width > m or y + height > n:
                                print('==========Error')
                                print(f"x = {x}, y = {y}, width = {width}, height = {height} Image size = {img.shape}")
                            haarFeature = HaarLikeFeature(x,y,width,height,haarType,threshold)
                            haarFeatures.append(haarFeature)
                            # count += 1
            # print(f"Feature type = {haarType.name}, count = {count}")

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
    return np.cumsum(np.cumsum(np.pad(img,((1,0),(1,0)),'constant'),axis=0),axis=1) # type: ignore

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


    #zero mean all images
    positiveImgs = positiveImgs - np.mean(positiveImgs)
    negativeImgs = negativeImgs - np.mean(negativeImgs)


    #unit variance all images
    positiveImgs = positiveImgs/np.var(positiveImgs)
    negativeImgs = negativeImgs/np.var(negativeImgs)


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
    def learn(self,positiveImgs,negativeImgs,threshold,maxFeatureWidth,maxFeatureHeight, nClassifiers):
        # positiveImgs : list of positive images
        # negativeImgs : list of negative images
        # threshold : threshold value
        # maxFeatureWidth : maximum width of a feature
        # maxFeatureHeight : maximum height of a feature
        
        #calculate integral images of positive and negative images
        print("Calculating integral images...")
        positiveIntegralImages = [integralImage(img) for img in positiveImgs]
        negativeIntegralImages = [integralImage(img) for img in negativeImgs]
        print("Done!")
        
        #calculate number of positive and negative samples
        nPositive = len(positiveImgs)
        nNegative = len(negativeImgs)
        
        #calculate number of features
        nImages = nPositive + nNegative
        
        #calculate number of rows and columns of the images
        height,width = positiveImgs[0].shape
        
        maxFeatureWidth = min(maxFeatureWidth,width)
        maxFeatureHeight = min(maxFeatureHeight,height)
        
        #create inital weights of samples and labels
        print("Creating initial weights and labels...")
        weightPositive = np.ones(nPositive)/(2 * nPositive)
        weightNegative = np.ones(nNegative)/(2 * nNegative)
        
        weights = np.concatenate((weightPositive,weightNegative))
        
        labels = np.concatenate((np.ones(nPositive),-np.ones(nNegative)))
        print("Done!")
        
        integralImages = positiveIntegralImages + negativeIntegralImages
        
        #calculate all possible haar features
        print("Calculating all possible haar features...")
        haarFeatures = determineFeatures(positiveImgs[0],threshold,maxFeatureWidth,maxFeatureHeight)
        print("Done!")
        
        #calculate number of haar features
        nHaarFeatures = len(haarFeatures)
        
        #create feature matrix
        votes = np.zeros((nImages,nHaarFeatures))
        
        #calculate votes of all haar features for all samples
        print("Calculating votes of all haar features for all samples...")
        #multiprocessing
        with Pool() as p:
            votes = p.starmap(getVotes,[(haarFeatures,integralImages[i]) for i in range(nImages)])
        votes = np.array(votes)


        # for i in range(nImages):
        #     votes[i,:] = np.array([haarFeatures[j].getVote(integralImages[i]) for j in range(nHaarFeatures)])

        # if votes1.all() == votes.all():
        #     print("Multiprocessing works!")
        # else:
        #     print("Multiprocessing doesn't work!")
        print("Done!")
                
        #select classifiers

        classifiers = []
        featureIndices:List[int] = list(range(nHaarFeatures))
        print(f"Training {nHaarFeatures} classifiers...")
        for i in range(nClassifiers):
            
            print("\r", f"Training classifier {i+1} of {nClassifiers}...", end="")

            classificationError = np.zeros(len(featureIndices))
            
            #normALIZE WEIGHTS
            weights = weights/np.sum(weights)
            
            #select classifier with minimum weighted error  
            for j in range(len(featureIndices)):
                #calculate classification error
                classificationError[j] = np.sum(weights[labels!=votes[:,featureIndices[j]]])
                
            minErrorIndex = np.argmin(classificationError)
            bestError = classificationError[minErrorIndex]
            bestFeatureIndex = featureIndices[minErrorIndex]
            
            #set feature weight
            bestFeature = haarFeatures[bestFeatureIndex]
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

            
def getVotes(haarFeatures,integralImage):
    return np.array([haarFeature.getVote(integralImage) for haarFeature in haarFeatures])
#================================================================================================
#================================================================================================