import sys
import os as os
from PyQt5 import QtWidgets
from tkinter import *
from tkinter import filedialog
from PyQt5.QtWidgets import *

from PyQt5.QtWidgets import QFileDialog, QDialog
# from definitions import ROOT_DIR
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap

def recognize():
    #call integration file and send to it the image path
    pass


image_path = ''
def browseFiles():
    image_path = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          )
                         


def window():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(20, 50, 400, 180)
    win.setWindowTitle("GUI")

    b1 = QtWidgets.QPushButton(win)
    b1.setText("Recognize")
    b1.setGeometry(50, 70, 110, 35)
    b1.clicked.connect(recognize)

    b2 = QtWidgets.QPushButton(win)
    b2.setText("Browse an Image")
    b2.setGeometry(230, 70, 125, 37)
    b2.clicked.connect(browseFiles)

    win.show()
    sys.exit(app.exec_())


window()
