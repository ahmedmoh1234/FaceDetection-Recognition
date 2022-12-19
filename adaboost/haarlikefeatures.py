import numpy as np
import enum

class HaarLikeFeature():

    class HaarType(enum.Enum):
        TWO_VERTICAL = (1,2)
        TWO_HORIZONTAL = (2,1)
        THREE_VERTICAL = (1,3)
        THREE_HORIZONTAL = (3,1)
        FOUR_DIAGONAL = (2,2)
    
    def __init__(self, x, y, width, height, haarType : HaarType, threshold , polarity) -> None:
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

        # polarity : polarity of the feature (1 or -1)
        self.polarity = polarity
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

        for haarType in HaarLikeFeature.HaarType:
            featureWidthStart = haarType.value[0]
            for width in range(featureWidthStart,maxFeatureWidth,haarType.value[0]):
                featureHeightStart = haarType.value[1]
                for height in range(featureHeightStart,maxFeatureHeight,haarType.value[1]):
                    for x in range(m - width):
                        for y in range(n - height):
                            for polarity in [-1,1]:
                                haarFeature = HaarLikeFeature(x,y,width,height,haarType,threshold,polarity)
                                haarFeatures.append(haarFeature)
        return haarFeatures

