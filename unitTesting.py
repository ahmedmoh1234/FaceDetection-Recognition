import numpy as np
import time
import adaboost.adaboost as adaboost
import adaboost.haarlikefeatures as HaarLikeFeatures

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
        intImg = np.cumsum(np.cumsum(np.pad(img,((1,0),(1,0)),'constant'),axis=0),axis=1)
        intImg2 = adaboost.integralImage(img)
        if intImg.all() == intImg2.all():
            print('Integral image test PASSED')


    def test_determineFeatures(self) -> None:
        # img : input image
        # n : number of rows
        # m : number of columns
        # intImg : integral image
        # features : list of features

        #calculate integral image with first row and first column as zeros
        #because this is a convention
        
        img = np.ones((384,288))
        # print(img.shape)
        print('Determining features...')
        t1 = time.time()
        hlf = HaarLikeFeatures.HaarLikeFeature(0,0,24,24,HaarLikeFeatures.HaarLikeFeature.HaarType.TWO_VERTICAL,0,1)
        features = hlf.determineFeatures(img,0,24,24)
        t2 = time.time()
        print('Number of features: ',len(features))
        print('Time taken to determine features: ',t2-t1)