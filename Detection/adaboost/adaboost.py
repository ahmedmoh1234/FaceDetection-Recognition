import numpy as np
from typing import List,Tuple
from functools import partial
from skimage.color import rgb2gray
from multiprocessing import Pool

import utilitis as ut




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
        positiveIntegralImages = [ut.integralImage(img) for img in positiveImgs]
        negativeIntegralImages = [ut.integralImage(img) for img in negativeImgs]
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
        haarFeatures = ut.determineFeatures(positiveImgs[0],threshold,minFeatureWidth,minFeatureHeight,maxFeatureWidth,maxFeatureHeight)
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
            votes[i,:] = np.array(list(pool.map(partial(ut.getFeatureVote, integralImage=integralImages[i]), haarFeatures)))


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


#================================================================================================