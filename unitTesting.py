import numpy as np
import adaboost.adaboost as adaboost

class UnitTest():

    def __init__(self, img) -> None:
        self.test_integralImage(img)
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