import numpy as np
import time

# import adaboost.adaboost as adaboost
import adaboost.haarlikefeatures as HLF
import adaboost.adaboost as ab



class UnitTest():

    def __init__(self, img) -> None:
        self.test_integralImage(img)
        self.test_determineFeatures()
        return

    def test_integralImage(self, img) -> None:

        # img : input image
        # n : number of rows
        # m : number of columns
        # intImg : integral image

        #calculate integral image with first row and first column as zeros
        #because this is a convention
        intImg = np.cumsum(np.cumsum(np.pad(img,( (1,0),(1,0) ), 'constant'),axis=0),axis=1) # type: ignore       
        intImg2 = ab.integralImage(img)
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
        hlf = HLF.HaarLikeFeature(0,0,24,24,HLF.HaarLikeFeature.HaarType.TWO_VERTICAL,0)
        features = ab.determineFeatures(img,0,24,24)
        if len(features) == 162336:
            print('Determine features test PASSED')
        else:
            #print with comas
            print(f'Determine features test FAILED. No of features : {len(features):,}')