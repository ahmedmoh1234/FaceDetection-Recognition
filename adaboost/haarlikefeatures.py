import numpy as np
import enum
import math
class HaarLikeFeature():

    class HaarType(enum.Enum):
        TWO_VERTICAL = (1,2)
        TWO_HORIZONTAL = (2,1)
        THREE_VERTICAL = (1,3)
        THREE_HORIZONTAL = (3,1)
        FOUR_DIAGONAL = (2,2)
    
    def __init__(self, x, y, width, height, haarType : HaarType, threshold) -> None:
        # x : x coordinate of top left corner of the feature
        self.x = x

        # y : y coordinate of top left corner of the feature
        self.y = y

        # width : width of the feature
        self.width = width

        # height : height of the feature
        self.height = height

        # haarType : type of haar like feature
        self.haarType = haarType

        # threshold : threshold value
        self.threshold = threshold
        
        return

    def determineFeatures(self, img, threshold, maxFeatureWidth, maxFeatureHeight) :
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
        for haarType in HaarLikeFeature.HaarType:
            featureWidthStart = haarType.value[0]
            for width in range(featureWidthStart,maxFeatureWidth + 1,haarType.value[0]):
                featureHeightStart = haarType.value[1]
                for height in range(featureHeightStart,maxFeatureHeight + 1,haarType.value[1]):
                    for x in range(m - width + 1):
                        for y in range(n - height + 1):
                            haarFeature = HaarLikeFeature(x,y,width,height,haarType,threshold)
                            haarFeatures.append(haarFeature)
        return haarFeatures


    #function to calculate sum of pixels using the integral image
    def calculateSum(intImg, x, y, width, height) : 
        # intImg : integral image
        # x : x coordinate of top left corner of the feature
        # y : y coordinate of top left corner of the feature
        # width : width of the feature
        # height : height of the feature
        # sum : sum of pixels
        sum = intImg[y + height, x + width] - intImg[y + height, x] - intImg[y, x + width] + intImg[y, x]
        return sum

    #function to calculate the value of the haar like feature
    def calculateFeatureValue(self, intImg) :
        # intImg : integral image
        # haarType : type of haar like feature
        # x : x coordinate of top left corner of the feature
        # y : y coordinate of top left corner of the feature
        # width : width of the feature
        # height : height of the feature
        # value : value of the haar like feature
        haarType = self.haarType
        x = self.x
        y = self.y
        width = self.width
        height = self.height
        if haarType == HaarLikeFeature.HaarType.TWO_VERTICAL:
            sum1 = HaarLikeFeature.calculateSum(intImg, x, y, width, height)
            sum2 = HaarLikeFeature.calculateSum(intImg, x, y + height, width, height)
            value = sum1 - sum2
        elif haarType == HaarLikeFeature.HaarType.TWO_HORIZONTAL:
            sum1 = HaarLikeFeature.calculateSum(intImg, x, y, width, height)
            sum2 = HaarLikeFeature.calculateSum(intImg, x + width, y, width, height)
            value = sum1 - sum2
        elif haarType == HaarLikeFeature.HaarType.THREE_VERTICAL:
            sum1 = HaarLikeFeature.calculateSum(intImg, x, y, width, height)
            sum2 = HaarLikeFeature.calculateSum(intImg, x, y + height, width, height)
            sum3 = HaarLikeFeature.calculateSum(intImg, x, y + 2*height, width, height)
            value = sum1 - sum2 + sum3
        elif haarType == HaarLikeFeature.HaarType.THREE_HORIZONTAL:
            sum1 = HaarLikeFeature.calculateSum(intImg, x, y, width, height)
            sum2 = HaarLikeFeature.calculateSum(intImg, x + width, y, width, height)
            sum3 = HaarLikeFeature.calculateSum(intImg, x + 2*width, y, width, height)
            value = sum1 - sum2 + sum3
        elif haarType == HaarLikeFeature.HaarType.FOUR_DIAGONAL:
            sum1 = HaarLikeFeature.calculateSum(intImg, x, y, width, height)
            sum2 = HaarLikeFeature.calculateSum(intImg, x + width, y, width, height)
            sum3 = HaarLikeFeature.calculateSum(intImg, x, y + height, width, height)
            sum4 = HaarLikeFeature.calculateSum(intImg, x + width, y + height, width, height)
            value = sum1 - sum2 - sum3 + sum4
            
        return value
    
    #function to calculate the polarity of the haar like feature
    def calculatePolarity(self, intImg) :
        
        # intImg : integral image
        # haarType : type of haar like feature
        # x : x coordinate of top left corner of the feature
        # y : y coordinate of top left corner of the feature
        # width : width of the feature
        # height : height of the feature
        # value : value of the haar like feature
        # polarity : polarity of the feature (1 or -1)
        value = self.calculateFeatureValue(intImg)
        if value >= self.threshold:
            polarity = 1
        else:
            polarity = -1
        return polarity
    
    #function to calculate the error of the haar like feature
    def calculateError(self, intImg, labels) :
        # intImg : integral image
        # labels : labels of the training samples
        # haarType : type of haar like feature
        # x : x coordinate of top left corner of the feature
        # y : y coordinate of top left corner of the feature
        # width : width of the feature
        # height : height of the feature
        # value : value of the haar like feature
        # polarity : polarity of the feature (1 or -1)
        # error : error of the feature
        polarity = self.calculatePolarity(intImg)
        error = 0
        for i in range(len(labels)):
            if labels[i] != polarity:
                error = error + 1
        return error
    
    #function to calculate the weighted error of the haar like feature
    def calculateWeightedError(self, intImg, labels, weights) :
        # intImg : integral image
        # labels : labels of the training samples
        # weights : weights of the training samples
        # haarType : type of haar like feature
        # x : x coordinate of top left corner of the feature
        # y : y coordinate of top left corner of the feature
        # width : width of the feature
        # height : height of the feature
        # value : value of the haar like feature
        # polarity : polarity of the feature (1 or -1)
        # error : error of the feature
        # weightedError : weighted error of the feature
        polarity = self.calculatePolarity(intImg)
        weightedError = 0
        for i in range(len(labels)):
            if labels[i] != polarity:
                weightedError = weightedError + weights[i]
        return weightedError
    
    #function to update the weights of the training samples
    def updateWeights(self, intImg, labels, weights, alpha) :
        # intImg : integral image
        # labels : labels of the training samples
        # weights : weights of the training samples
        # alpha : alpha value of the feature
        # haarType : type of haar like feature
        # x : x coordinate of top left corner of the feature
        # y : y coordinate of top left corner of the feature
        # width : width of the feature
        # height : height of the feature
        # value : value of the haar like feature
        # polarity : polarity of the feature (1 or -1)
        # error : error of the feature
        # weightedError : weighted error of the feature
        # newWeights : new weights of the training samples
        polarity = self.calculatePolarity(intImg)
        newWeights = []
        for i in range(len(labels)):
            if labels[i] != polarity:
                newWeights.append(weights[i] * math.exp(alpha))
            else:
                newWeights.append(weights[i] * math.exp(-alpha))
        return newWeights
    
    #function to calculate the alpha value of the haar like feature
    def calculateAlpha(self, intImg, labels, weights) :
        
        # intImg : integral image
        # labels : labels of the training samples
        # weights : weights of the training samples
        # haarType : type of haar like feature
        # x : x coordinate of top left corner of the feature
        # y : y coordinate of top left corner of the feature
        # width : width of the feature
        # height : height of the feature
        # value : value of the haar like feature
        # polarity : polarity of the feature (1 or -1)
        # error : error of the feature
        # weightedError : weighted error of the feature
        # alpha : alpha value of the feature
        weightedError = self.calculateWeightedError(intImg, labels, weights)
        alpha = 0.5 * math.log((1 - weightedError) / weightedError)
        return alpha
    
    
    