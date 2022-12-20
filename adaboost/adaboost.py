import numpy as np
import haarlikefeatures as hlf
from typing import List,Tuple
#================================================================================================
#================================================================================================
def determineFeatures(img, threshold, maxFeatureWidth, maxFeatureHeight) -> List[hlf.HaarLikeFeature]:
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
        count = 0
        for haarType in hlf.HaarLikeFeature.HaarType:
            featureWidthStart = haarType.value[0]
            for width in range(featureWidthStart,maxFeatureWidth + 1,haarType.value[0]):
                featureHeightStart = haarType.value[1]
                for height in range(featureHeightStart,maxFeatureHeight + 1,haarType.value[1]):
                    for x in range(m - width + 1):
                        for y in range(n - height + 1):
                            haarFeature = hlf.HaarLikeFeature(x,y,width,height,haarType,threshold)
                            haarFeatures.append(haarFeature)
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
    return np.cumsum(np.cumsum(np.pad(img,((1,0),(1,0)),'constant'),axis=0),axis=1)

#================================================================================================
#================================================================================================

class DecisionStump:
    # Stump is a weak classifier
    # It is a decision tree with only one split (depth = 1)
    # It is used to classify samples into two classes
    def __init__(self):
        return

    def classificationError(self,y,yPred):
        # y : target variable
        # yPred : predicted value
        return len(y[y!=yPred])/len(y)

    def predict(self,X):
        # X : feature matrix
        # n : number of samples
        # m : number of features
        return

        

#================================================================================================
#================================================================================================

class AdaBoost:

    #weakClassifier : weak classifier (Gm)
    #M : number of boosting iterations

    def __init__(self):
        self.alphas = []
        self.weakClassifiers = []
        self.M = None
        self.trainingErrors = []
        self.predictionErrors = []

    #learn tak
    def learn(positiveImgs,negativeImgs,threshold,maxFeatureWidth,maxFeatureHeight):
        # positiveImgs : list of positive images
        # negativeImgs : list of negative images
        # threshold : threshold value
        # maxFeatureWidth : maximum width of a feature
        # maxFeatureHeight : maximum height of a feature
        
        #calculate integral images of positive and negative images
        positiveIntegralImages = [integralImage(img) for img in positiveImgs]
        negativeIntegralImages = [integralImage(img) for img in negativeImgs]
        
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
        weightPositive = np.ones(nPositive)/(2 * nPositive)
        weightNegative = np.ones(nNegative)/(2 * nNegative)
        
        weights = np.concatenate((weightPositive,weightNegative))
        
        labels = np.concatenate((np.ones(nPositive),-np.ones(nNegative)))
        
        integralImages = positiveIntegralImages + negativeIntegralImages
        
        #calculate all possible haar features
        haarFeatures = determineFeatures(integralImages[0],threshold,maxFeatureWidth,maxFeatureHeight)
        
        #calculate number of haar features
        nHaarFeatures = len(haarFeatures)
        
        #create feature matrix
        votes = np.zeros((nImages,nHaarFeatures))
        
        #calculate votes of all haar features for all samples
        for i in range(nImages):
            votes[i,:] = np.array([haarFeatures[j].getVote(integralImages[i]) for j in range(nHaarFeatures)])
                
        #select classifiers

        classifiers = []
        featureIndices:List[int] = List(range(nHaarFeatures))
        for i in range(nHaarFeatures):
            
            classificationError = np.zeros(len(nHaarFeatures))
            
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

            


#================================================================================================
#================================================================================================