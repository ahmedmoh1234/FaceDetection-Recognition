Face Detection:

Face detection requires the following libraries (Python modules are not included):
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


Face Recognition:

Face recognition required the followgin libraries:
	- Numpy
	- sklearn
	- matplotlib
	- Random
	- cv2
	- os
	- Skimmage
	- PyQt5

We created the following files:
	- main_recognizer.py
	- recognizer.ipynb

In order to run the project, go to recognition_main.py and do the following
1- Call the function Train Model
2- Call the function main_recognizer with the image you want to recognize
3- If the person does not exist in the training dataset then it prints unknown else, it shows the image of the nearest person
4- if you want to run the Olivetti Dataset on 100 trial to show the accuracy, run all the cells in recognizer.ipynb

