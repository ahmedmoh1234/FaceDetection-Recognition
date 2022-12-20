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

        # weight : weight of the feature
        self.weight = 0
        
        return

   


    

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
            sum1 = calculateSum(intImg, x, y, width, height)
            sum2 = calculateSum(intImg, x, y + height, width, height)
            value = sum1 - sum2
        elif haarType == HaarLikeFeature.HaarType.TWO_HORIZONTAL:
            sum1 = calculateSum(intImg, x, y, width, height)
            sum2 = calculateSum(intImg, x + width, y, width, height)
            value = sum1 - sum2
        elif haarType == HaarLikeFeature.HaarType.THREE_VERTICAL:
            sum1 = calculateSum(intImg, x, y, width, height)
            sum2 = calculateSum(intImg, x, y + height, width, height)
            sum3 = calculateSum(intImg, x, y + 2*height, width, height)
            value = sum1 - sum2 + sum3
        elif haarType == HaarLikeFeature.HaarType.THREE_HORIZONTAL:
            sum1 = calculateSum(intImg, x, y, width, height)
            sum2 = calculateSum(intImg, x + width, y, width, height)
            sum3 = calculateSum(intImg, x + 2*width, y, width, height)
            value = sum1 - sum2 + sum3
        elif haarType == HaarLikeFeature.HaarType.FOUR_DIAGONAL:
            sum1 = calculateSum(intImg, x, y, width, height)
            sum2 = calculateSum(intImg, x + width, y, width, height)
            sum3 = calculateSum(intImg, x, y + height, width, height)
            sum4 = calculateSum(intImg, x + width, y + height, width, height)
            value = sum1 - sum2 - sum3 + sum4
        else:
            print("=====INVALID Haar Type")
            value = None
            
        return value
    
    #function to calculate the polarity of the haar like feature
    def getVote(self, intImg) :
        # intImg : integral image
        # haarType : type of haar like feature
        # x : x coordinate of top left corner of the feature
        # y : y coordinate of top left corner of the feature
        # width : width of the feature
        # height : height of the feature
        # threshold : threshold value
        # value : value of the haar like feature
        # polarity : polarity of the haar like feature
        value = self.calculateFeatureValue(intImg)
        threshold = self.threshold
        if value >= threshold:
            vote = 1
        else:
            vote = -1
        return vote

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