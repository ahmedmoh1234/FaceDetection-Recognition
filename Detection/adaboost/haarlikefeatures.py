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
    
    def __init__(self, x, y, width, height, haarType : HaarType, threshold, polarity) -> None:
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
        self.weight = 1

        # polarity : polarity of the feature
        self.polarity = polarity
        
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
        if (x + width >= intImg.shape[1] or y + height >= intImg.shape[0]):
            print('Error1: Feature out of bounds')
            print(f"intImg shape: {intImg.shape}")
        if haarType == HaarLikeFeature.HaarType.TWO_VERTICAL:
            midHeight = math.floor(height/2)
            sum1 = self.calculateSum(intImg, x, y            , width, midHeight)
            sum2 = self.calculateSum(intImg, x, y + midHeight, width, midHeight)
            value = sum1 - sum2
        elif haarType == HaarLikeFeature.HaarType.TWO_HORIZONTAL:
            midWidth = math.floor(width/2)
            sum1 = self.calculateSum(intImg, x           , y, midWidth, height)
            sum2 = self.calculateSum(intImg, x + midWidth, y, midWidth, height)
            value = sum1 - sum2
        elif haarType == HaarLikeFeature.HaarType.THREE_VERTICAL:
            h1 = math.floor(height/3)
            h2 = math.floor(2*height/3)
            sum1 = self.calculateSum(intImg, x, y        , width, h1)
            sum2 = self.calculateSum(intImg, x, y + h1   , width, h1)
            sum3 = self.calculateSum(intImg, x, y + h2   , width, h1)
            value = sum1 - sum2 + sum3
        elif haarType == HaarLikeFeature.HaarType.THREE_HORIZONTAL:
            w1 = math.floor(width/3)
            w2 = math.floor(2*width/3)
            sum1 = self.calculateSum(intImg, x       , y , w1, height)
            sum2 = self.calculateSum(intImg, x + w1  , y , w1, height)
            sum3 = self.calculateSum(intImg, x + w2  , y , w1, height)
            value = sum1 - sum2 + sum3
        elif haarType == HaarLikeFeature.HaarType.FOUR_DIAGONAL:
            h1 = math.floor(height/2)
            w1 = math.floor(width/2)

            sum1 = self.calculateSum(intImg, x       , y     , w1, h1)
            sum2 = self.calculateSum(intImg, x + w1  , y     , w1, h1)
            sum3 = self.calculateSum(intImg, x       , y + h1, w1, h1)
            sum4 = self.calculateSum(intImg, x + w1  , y + h1, w1, h1)
            value = sum1 - sum2 - sum3 + sum4
        else:
            print("=====INVALID Haar Type")
            value = None
            
        return value
    
    #function to calculate the polarity of the haar like feature
    def getVote(self, intImg) -> float:
        # intImg : integral image
        # haarType : type of haar like feature
        # x : x coordinate of top left corner of the feature
        # y : y coordinate of top left corner of the feature
        # width : width of the feature
        # height : height of the feature
        # threshold : threshold value
        # value : value of the haar like feature
        # polarity : polarity of the haar like feature

        # print(f"getvote int image shape: {intImg.shape}")

        value = self.calculateFeatureValue(intImg) * self.polarity
        threshold = self.threshold
        if value >= threshold:
            vote = 1 * self.weight
        else:
            vote = -1 * self.weight
        return vote

    #function to calculate sum of pixels using the integral image
    def calculateSum(self, intImg, x, y, width, height) : 
        # intImg : integral image
        # x : x coordinate of top left corner of the feature
        # y : y coordinate of top left corner of the feature
        # width : width of the feature
        # height : height of the feature
        # sum : sum of pixels
        x = x + 1
        y = y + 1
        height = height - 1
        width = width - 1
        if (x + width >= intImg.shape[1] or y + height >= intImg.shape[0]):
            print('Error2: Feature out of bounds')
            print(f"x: {x}, y: {y}, width: {width}, height: {height}, intImg.shape: {intImg.shape}")
        sum = intImg[y + height, x + width] - intImg[y + height, x - 1] - intImg[y - 1, x + width] + intImg[y - 1, x - 1]
        return sum

