import numpy as np
import skimage.io as io
import os
import sys
sys.path.append('Recognizer/Eigenfaces')
import main_recognizer
sys.path.append('Detection')
import DetectionMain

dirname = os.path.abspath(os.getcwd())
test_images_path = os.path.join(dirname, 'Test Images')


def main(image_path):
    classifiersToBeUsed = DetectionMain.trainDetector()
    # img = io.imread(test_images_path + '/Mostafa_1.jpeg')
    img = io.imread(image_path)
    detectedFace = DetectionMain.detector_main(img , classifiersToBeUsed)
    if(detectedFace):
        # Recognize the face
        main_recognizer.recognizer_main(img)

    # img = io.imread(test_images_path + '/1.png')

    # detectedFace = DetectionMain.detector_main(img, classifiersToBeUsed)
    # if(detectedFace):
    #     # Recognize the face
    #     main_recognizer.recognizer_main(img)


    # img = io.imread(test_images_path + '/Apple.jpg')

    # detectedFace = DetectionMain.detector_main(img, classifiersToBeUsed)
    # if(detectedFace):
    #     # Recognize the face
    #     main_recognizer.recognizer_main(img)


def train():
    # Train the detector
    DetectionMain.trainDetector()
    # Train the recognizer
    main_recognizer.train_our()


if __name__ == '__main__':
    # train()
    main()
