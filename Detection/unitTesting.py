import numpy as np

# import adaboost.adaboost as adaboost
import adaboost.haarlikefeatures as HLF
import utilitis as ut



class UnitTest():

    def __init__(self, img) -> None:
        self.test_integralImage(img)
        self.test_determineFeatures()
        self.test_calculateSum()
        self.test_featureValue()
        return

    def test_integralImage(self, img) -> None:

        # img : input image
        # n : number of rows
        # m : number of columns
        # intImg : integral image

        #calculate integral image with first row and first column as zeros
        #because this is a convention
        intImg = np.zeros((img.shape[0] + 1, img.shape[1] + 1))
        for i in range(1, img.shape[0] + 1):
            for j in range(1, img.shape[1] + 1):
                intImg[i][j] = intImg[i][j - 1] + intImg[i - 1][j] - intImg[i - 1][j - 1] + img[i - 1][j - 1]
        intImg2 = ut.integralImage(img)
        if intImg.all() == intImg2.all():
            print('Integral image test PASSED')
        else:
            print('Integral image test FAILED')


    def test_determineFeatures(self) -> None:

        # img : input image
        # n : number of rows
        # m : number of columns
        # intImg : integral image
        # features : list of features

        #calculate integral image with first row and first column as zeros
        #because this is a convention
        
        img = np.ones((24,24))
        # print(img.shape)
        # hlf = HLF.HaarLikeFeature(0,0,24,24,HLF.HaarLikeFeature.HaarType.TWO_VERTICAL,0, 1)
        features = ut.determineFeatures(img,0,0,0,24,24)
        if len(features) == 162336 * 2:
            print('Determine features test PASSED')
        else:
            #print with comas
            print(f'Determine features test FAILED. No of features : {len(features):,}')

    
    def test_calculateSum(self) -> None:

        # intImg : integral image
        # x : x coordinate of top left corner of the feature
        # y : y coordinate of top left corner of the feature
        # width : width of the feature
        # height : height of the feature
        # sum : sum of the feature
    
        orgImage = np.ones((4,4))
        orgImage[1:2,1:3] = 2
        orgImage[2:3,2:3] = 3
        # orgImage now looks like this:
        # 1. 1. 1. 1.
        # 1.(2. 2.)1.
        # 1.(1. 3.)1.
        # 1. 1. 1. 1.

        x = 1
        y = 1
        width = 2
        height = 2

        targetSum = 8

        intImg = ut.integralImage(orgImage)
        #The output of integralImage is that
        #   0.  0.  0.  0.  0.
        #   0.  1.  2.  3.  4.
        #   0.  2. (5.  8.)10.
        #   0.  3. (7. 13.)16.
        #   0.  4.  9. 16. 20.

        tempFeature = HLF.HaarLikeFeature(x, y, width, height, HLF.HaarLikeFeature.HaarType.TWO_VERTICAL, 0, 1)
        sum = tempFeature.calculateSum(intImg, x, y, width, height)
        if sum == targetSum:
            print('Calculate sum test 1 PASSED')
        else:
            print(f"Calculate sum test 1 FAILED. Sum : {sum} , targetSum : {targetSum} ")



        orgImg = np.ones((10,10))
        orgImg[4:8,4:8] = 2
        # orgImg now looks like this:
        # 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
        # 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
        # 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
        # 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
        # 1. 1. 1. 1. 2. 2. 2. 2. 1. 1.
        # 1. 1. 1. 1. 2. 2. 2. 2. 1. 1.
        # 1. 1. 1. 1.(2. 2. 2. 2.)1. 1.
        # 1. 1. 1. 1.(2. 2. 2. 2.)1. 1.
        # 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
        # 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.

        x = 4
        y = 6
        width = 4
        height = 2

        intImg = ut.integralImage(orgImg)
        # intImg now looks like this:
        #   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
        #   0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.
        #   0.   2.   4.   6.   8.  10.  12.  14.  16.  18.  20.
        #   0.   3.   6.   9.  12.  15.  18.  21.  24.  27.  30.
        #   0.   4.   8.  12.  16.  20.  24.  28.  32.  36.  40.
        #   0.   5.  10.  15.  20.  26.  32.  38.  44.  49.  54.
        #   0.   6.  12.  18.  (24.  32.  40.  48.  56.  62.)  68.
        #   0.   7.  14.  21.  (28.  38.  48.  58.  68.  75.)  82.
        #   0.   8.  16.  24.   32.  44.  56.  68.  80.  88.  96.
        #   0.   9.  18.  27.   36.  49.  62.  75.  88.  97. 106.
        #   0.  10.  20.  30.   40.  54.  68.  82.  96. 106. 116.

        targetSum = 16

        sum = tempFeature.calculateSum(intImg, x, y, width, height)
        if sum == targetSum:
            print('Calculate sum test 2 PASSED')
        else:
            print(f"Calculate sum test 2 FAILED. Sum : {sum}, target sum : {targetSum}")


        orgImg = np.ones((5,5))
        orgImg[1:3,1:3] = 4
        orgImg[2:4,2:4] = 7
        # orgImg now looks like this:
        # 1. 1. 1. 1. 1.
        # 1.(4. 4.)1. 1.
        # 1. 4. 7. 7. 1.
        # 1.(1. 7.)7. 1.
        # 1. 1. 1. 1. 1.

        x = 1
        y = 1
        width = 2
        height = 3

        targetSum = 27

        intImg = ut.integralImage(orgImg)
        # intImg now looks like this:
        #  0.  0.  0.  0.  0.  0.
        #  0. (1.  2.) 3.  4.  5.
        #  0.  2.  7. 12. 14. 16.
        #  0. (3. 12.)24. 33. 36.
        #  0.  4. 14. 33. 49. 53.
        #  0.  5. 16. 36. 53. 58.

        sum = tempFeature.calculateSum(intImg, x, y, width, height)
        if sum == targetSum:
            print('Calculate sum test 3 PASSED')
        else:
            print(f"Calculate sum test 3 FAILED. Sum : {sum}, target sum : {targetSum}")


    def test_featureValue(self) -> None:
        testImg = np.ones((5,5))
        testImg[1:3,1:3] = 4
        testImg[2:4,2:4] = 7
        # testImg now looks like this:
        # 1. 1. 1. 1. 1.
        # 1.(4. 4. 1.)1.
        # 1. 4. 7. 7. 1.
        # 1.(1. 7. 7.)1.
        # 1. 1. 1. 1. 1.

        x = 1
        y = 1
        width = 3
        height = 3

        intImg = ut.integralImage(testImg)

        hFeature = HLF.HaarLikeFeature(x, y, width, height, HLF.HaarLikeFeature.HaarType.THREE_VERTICAL,0,1) 
        # hFeature now looks like this:
        # 0. 0. 0. 0. 0.
        # 0. 1. 1. 1. 0.
        # 0. 1. 1. 1. 0.
        # 0. 1. 1. 1. 0.
        # 0. 0. 0. 0. 0.

        featureValue = hFeature.calculateFeatureValue(intImg)
        targetValue = 6
        if featureValue == None:
            print('Feature value test FAILED. Invalid feature type')
        elif featureValue == targetValue:
            print('Feature value test 1 PASSED')
        else:
            print(f"Feature value test 1 FAILED. Feature value : {featureValue}, target value : {targetValue}")

        hFeature = HLF.HaarLikeFeature(x, y, width-1, height, HLF.HaarLikeFeature.HaarType.THREE_VERTICAL,0,1)

        featureValue = hFeature.calculateFeatureValue(intImg)
        targetValue = 5
        if featureValue == None:
            print('Feature value test FAILED. Invalid feature type')
        elif featureValue == targetValue:
            print('Feature value test 2 PASSED')
        else:
            print(f"Feature value test 2 FAILED. Feature value : {featureValue}, target value : {targetValue}")


        