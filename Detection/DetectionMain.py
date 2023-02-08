import cv2
import skimage.io as io
import skimage as sk
from skimage.color import rgb2gray
from skimage.transform import resize
import time
import numpy as np
import os
import sys
import copy
from tqdm.auto import tqdm

# from unitTesting import UnitTest
import adaboost.adaboost as ab
import utilitis as ut
_file_ = os.path.abspath(__file__)
dirname = os.path.dirname(_file_)


#===============================================Training ===============================================
def trainDetector():
    
    positiveDatasetPath = os.path.join(dirname, 'Dataset/posTrain2.npy')

    # Not needed for now
    # img = io.imread(test_filename)
    # img = rgb2gray(img)
    # unitTesting = UnitTest(img)


    # load positive dataset
    try:
        # print("Loading positive dataset...")
        positiveDataset = np.load(positiveDatasetPath)
    except:
        print("No dataset found, creating new dataset...")
        # positiveDataset = np.load("./Dataset/olivetti_faces.npy")
        positiveDataset = np.load("./testImg/postTrain.npy")
        personID = np.load("./Dataset/olivetti_faces_target.npy")

        # resize images to 24x24
        newPositiveDataset = np.empty(shape=(positiveDataset.shape[0], 24, 24))
        for i in range(len(positiveDataset)):
            newImg = np.array(positiveDataset[i])

            finalImg = resize(newImg, (24, 24))
            newPositiveDataset[i] = finalImg

        positiveDataset = newPositiveDataset

        # for i in range(10):
        #     io.imshow(positiveDataset[i])
        #     io.show()

        # print(positiveDataset.shape)

        # np.save("./Dataset/positiveDataset.npy", positiveDataset)
        # print("Positive Dataset saved")

    # loading negative dataset
    try:
        print("Loading negative dataset...")
        negativeDatasetPath = os.path.join(dirname, 'Dataset/negTrain.npy')
        negativeDataset = np.load(negativeDatasetPath)
    except:
        print("No dataset found, creating new dataset...")
        directory = "./Dataset/NegativeSet"
        negativeDataset = np.empty(shape=(len(os.listdir(directory)), 24, 24))
        i = 0
        for filename in os.listdir(directory):
            if filename.endswith(".jpg"):
                # test_filename = os.path.join(dirname, 'test.jpg')
                curImg = io.imread(os.path.join(directory, filename), as_gray=True)
                newImg = resize(curImg, (24, 24))
                negativeDataset[i] = newImg
                i += 1
        # negativeDataset = negativeDataset.reshape(len(os.listdir(directory)), 64, 64)
        # print(negativeDataset.shape)
        # np.save("./Dataset/negativeDataset.npy", negativeDataset)
        # print("Negative Dataset saved")


    posDataset, negDataset = ut.preprocessImages(positiveDataset, negativeDataset)

    # print(f"Positive Dataset Shape: {posDataset.shape}")
    # print(f"Negative Dataset Shape: {negDataset.shape}")

    minFeatureWidth = 4
    maxFeatureWidth = 20
    minFeatureHeight = 4
    maxFeatureHeight = 20
    nClassifiers = 5001
    threshold = 0

    classifiersToBeUsed = []
    try:
        # print("Loading classifiers...")
        classifiersToBeUsedPath = os.path.join(dirname, 'Classifiers/classifiers5001-0-4-20.npy')
        classifiersToBeUsed = np.load(classifiersToBeUsedPath, allow_pickle=True)
    except Exception as e:
        # print(e)
        print("No classifiers found, creating new classifiers...")
        classifier = ab.AdaBoost()
        start = time.time()
        classifiersToBeUsed = classifier.learn(
            posDataset, negDataset, threshold, minFeatureWidth, minFeatureHeight, maxFeatureWidth, maxFeatureHeight, nClassifiers)
        end = time.time()
        classifiersToBeUsed = np.array(classifiersToBeUsed)
        np.save("./classifiers"+str(nClassifiers)+"-" + str(threshold) + "-" +
                str(minFeatureWidth)+"-" + str(maxFeatureWidth) + ".npy", classifiersToBeUsed)
        print("Time taken to train the classifier: ", end-start)

    print("Classifiers loaded successfully")
    return classifiersToBeUsed

#===============================================Testing ===============================================
def testDetector(classifiersToBeUsed):
    testDataSetPath = os.path.join(dirname, 'Dataset/testMix.npy')
    testDataSet = np.load(testDataSetPath)
    testDataSetTargetPath = os.path.join(dirname, "Dataset/testTargets.npy")

    testDataSetTarget = np.load(testDataSetTargetPath)
    # convert to int
    testDataSetTarget = testDataSetTarget.astype(int)

    # print(f"Test Dataset Shape: {testDataSet.shape}")

    # get integral images
    ii = np.array([ut.integralImage(test) for test in testDataSet])

    i = 0
    correctCount = 0
    firstWrong = 5
    facesCount = 0
    correctFacesCount = 0
    correctNonFacesCount = 0
    nonFacesCount = 0
    for test in ii:
        predicted = ut.getVotes(np.array(classifiersToBeUsed), test)
        if predicted == testDataSetTarget[i]:
            correctCount += 1
            if predicted == 1:
                correctFacesCount += 1
            else:
                correctNonFacesCount += 1
        else:
            if firstWrong > 0:
                print("Predicted: ", predicted)
                print("Target: ", testDataSetTarget[i])
                # io.imshow(testDataSet[i])
                # io.show()
                firstWrong -= 1
        if testDataSetTarget[i] == 1:
            facesCount += 1
        if testDataSetTarget[i] == 0:
            nonFacesCount += 1

        i += 1

    print("Accuracy: ", correctCount/len(testDataSet) * 100, "%")
    print(f"Faces detected: {correctFacesCount} / {facesCount} , Accuracy: {correctFacesCount/facesCount * 100}%")
    print(f"Non Faces detected: {correctNonFacesCount} / {nonFacesCount} , Accuracy: {correctNonFacesCount/nonFacesCount * 100}%")


def scaleFeatures(features, scale):

    features2 = copy.deepcopy(features)
    try:
        finalFeaturesPath = os.path.join(
            dirname, 'ScaledFeatures/scaledFeatures'+str(scale) + '.npy')
        finalFeatures = np.load(
            finalFeaturesPath, allow_pickle=True)
        # print(f"Features with scale {scale} loaded successfully")
        return np.array(finalFeatures)
    except:
        print("No features found, creating new features...")

        finalFeatures = []
        for f in features2:
            if f.x + f.width > 24:
                print(f"x + width > 24, x: {f.x}, width: {f.width}")

            if f.y + f.height > 24:
                print(f"y + height > 24 , y: {f.y}, height: {f.height}")

            f = f * float(scale)
            finalFeatures.append(f)
        finalFeatures2 = np.array(finalFeatures)
        np.save("./scaledFeatures"+str(scale) + ".npy", finalFeatures2)
        print("Features saved successfully")
        return finalFeatures2


def predict(img2, classifiersToBeUsedFn) -> list:
    img = np.copy(img2)
    if len(img.shape) >2:
        img = img[:,:,:3]
        img = rgb2gray(img)
    img = img/img.max()
    img = img / np.var(img)

    ii = ut.integralImage(img)
    imgHeight, imgWidth = img.shape

    if imgHeight < 30 or imgWidth < 30:
        return []

    # scaling features
    maxScaleFactor = min(imgHeight/24, imgWidth/24)

    detectedFaces = []

    # print("Scaling features...")
    # scaledFeatures2_5 = scaleFeatures(classifiersToBeUsedFn, 2.5)
    # scaledFeatures2_5 = scaledFeatures2_5.squeeze()

    # scaledFeatures3_75 = scaleFeatures(classifiersToBeUsedFn, 3.75)
    # scaledFeatures3_75 = scaledFeatures3_75.squeeze()

    # scaledFeatures5 = scaleFeatures(classifiersToBeUsedFn, 5)
    # scaledFeatures5 = scaledFeatures5.squeeze()

    # scaledFeatures6_25 = scaleFeatures(classifiersToBeUsedFn, 6.25)
    # scaledFeatures6_25 = scaledFeatures6_25.squeeze()

    # scaledFeatures7_5 = scaleFeatures(classifiersToBeUsedFn, 7.5)
    # scaledFeatures7_5 = scaledFeatures7_5.squeeze()

    # scaledFeatures8_75 = scaleFeatures(classifiersToBeUsedFn, 8.75)
    # scaledFeatures8_75 = scaledFeatures8_75.squeeze()

    # scaledFeatures10 = scaleFeatures(classifiersToBeUsedFn, 10)
    # scaledFeatures10 = scaledFeatures10.squeeze()

    # scaledFeatures11_25 = scaleFeatures(classifiersToBeUsedFn, 11.25)
    # scaledFeatures11_25 = scaledFeatures11_25.squeeze()

    # scaledFeatures12_5 = scaleFeatures(classifiersToBeUsedFn, 12.5)
    # scaledFeatures12_5 = scaledFeatures12_5.squeeze()

    # print("Features scaled successfully\n")
    # scaledFeatures = [scaledFeatures2_5, scaledFeatures3_75, scaledFeatures5, scaledFeatures6_25,
    #                   scaledFeatures7_5, scaledFeatures8_75, scaledFeatures10, scaledFeatures11_25, scaledFeatures12_5]

    # print the max x and y in classifiersToBeUsed
    # maxX = 0
    # maxY = 0
    # maxWidth = 0
    # maxHeight = 0
    # maxTotX = 0
    # maxTotY = 0
    # for f in scaledFeatures5 :
    #     if not isinstance(f.x, int):
    #         print("x is float")
    #     if not isinstance(f.y, int):
    #         print("y is float")
    #     if not isinstance(f.width, int):
    #         print("width is float")
    #     if not isinstance(f.height, int):
    #         print("height is float")

    #     if f.x + f.width > maxTotX:
    #         maxX = f.x
    #         maxWidth = f.width
    #         maxTotX = f.x + f.width
    #     if f.y + f.height > maxTotY:
    #         maxY = f.y
    #         maxHeight = f.height
    #         maxTotY = f.y + f.height
    # print(f"===========Max x: {maxX} and maxWidth = {maxWidth} with MaxTotX = {maxTotX} and max y: {maxY} and maxHeight = {maxHeight} with MaxTotY = {maxTotY}")
    # print(f"scale factor: {scaledFeatures5[0].scale} \n")

    # print(f"shape of scaled features: {scaledFeatures[0].shape}")

    print("Starting to detect faces...")
    shiftValue = 1
    noOfScales = min(img.shape[0], img.shape[1]) // (30*1.1)      #How many times we will scale the image down using a scale factor of 1.1

    for scale in range(int(noOfScales)):
        newImg = cv2.resize(img, (int(imgWidth / 1.1), int(imgHeight / 1.1)))
        for x in tqdm(range(0, int(imgWidth - 24 ), shiftValue )):
            for y in range(0, int(imgHeight - 24 ), shiftValue ):
                # print(f"scale factor: {scaleFactor}, x: {x}, y: {y}")
                if ut.getVotes(classifiersToBeUsedFn, ii) == 1:
                    detectedFaces.append([x, y, 1])
        # print("Finished scale factor: ", scaleFactor)

    return detectedFaces



def detectFaces(img, classifiersToBeUsedFn, threshold=0):
    if img.shape[0] > 150 or img.shape[1] > 150:
        img = resize(img, (150, 150))
    detectedFaces = predict(img, classifiersToBeUsedFn)

    if len(detectedFaces) > threshold:
        print(f"==========Face detected==============")
        return img
    else:
        print("================No faces detected================")
        return None



def detector_main(input_image, classifiersToBeUsed):
    detectedFace = detectFaces(input_image, classifiersToBeUsed, 30)
    return detectedFace is not None


classifiersToBeUsed = np.load("Classifiers/classifiers.npy", allow_pickle=True)
img = cv2.imread("test.jpg")
detectFaces1 = detectFaces(img, classifiersToBeUsed, 30)

if detectFaces1 is not None:
    cv2.imshow("Detected face", detectFaces1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()