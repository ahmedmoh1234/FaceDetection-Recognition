import main_recognizer
import numpy as np
# import DetectionMain
import skimage.io as io
import os
_file_ = os.path.abspath(__file__)
dirname = os.path.dirname(_file_)


def main():
    imgPath = os.path.join(dirname, 'testImg.jpg')
    img = io.imread(imgPath)
    # detectedFace = DetectionMain.detector_main(img)
    if(True):
        # Recognize the face
        main_recognizer.recognizer_main(img)

    imgPath = os.path.join(dirname, 'TestImg/Lopez.jpg')
    img = io.imread(imgPath)
    # detectedFace = DetectionMain.detector_main(img)
    if(True):
        # Recognize the face
        main_recognizer.recognizer_main(img)

    imgPath = os.path.join(dirname, 'olivetti_faces_10.jpg')
    img = io.imread(imgPath)
    # detectedFace = DetectionMain.detector_main(img)
    if(True):
        # Recognize the face
        main_recognizer.recognizer_main(img)


def train():
    # Train the detector

    # Train the recognizer
    main_recognizer.train()


if __name__ == '__main__':
    main()
