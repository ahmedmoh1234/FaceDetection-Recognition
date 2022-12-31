import numpy as np
import skimage.io as io
import os
import sys
sys.path.append('Recognizer/Eigenfaces')
import main_recognizer
sys.path.append('Detection')
# import DetectionMain

dirname = os.path.abspath(os.getcwd())
test_images_path = os.path.join(dirname, 'Test Images')


def main():
    img = io.imread(test_images_path + '/testImg.jpg')
    # detectedFace = DetectionMain.detector_main(img)
    if(True):
        # Recognize the face
        main_recognizer.recognizer_main(img)

    img = io.imread(test_images_path + '/Lopez.jpg')

    # detectedFace = DetectionMain.detector_main(img)
    if(True):
        # Recognize the face
        main_recognizer.recognizer_main(img)

    img = io.imread(test_images_path + '/olivetti_faces_10.jpg')

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
