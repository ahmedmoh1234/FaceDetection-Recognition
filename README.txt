Face Detection:

Face detection requires the following libraries:
	- NumPy
	- skimage
	- tqdm (Progress bar)

We created the following files:
	- unitTesting.py : This was used at the beginning to make sure that separate functions were working
	- utilitis.py : This file contains the common functions such as integralImage
	- adaboost.py : This file implements AdaBoost
	- haarlikefeatures.py : This file implements Haar-like features

In order to run the project, go to DetectionMain.py and do the following:
1 - Run "trainDetector()" and save the return to "classifiersToBeUsed"
2 - Get the image you want to test it and save it to "img"
3 - Call "detector_main(img,classifiersToBeUsed)". Will return "True" if face was found, "None" otherwise.