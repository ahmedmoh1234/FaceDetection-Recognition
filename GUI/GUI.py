import sys
import os as os
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from tkinter import *
from tkinter import filedialog


def run():
    print('Run')


def test():
    return 10


def browseFiles():
    # filename = filedialog.askopenfilename(initialdir="/",
    #                                       title="Select a File",
    #                                       filetypes=(("Text files",
    #                                                   "*.txt*"),
    #                                                  ("all files",
    #                                                   "*.*")))
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          )
    # # Change label contents
    # label_file_explorer.configure(text="File Opened: "+filename)


def browseFiles2():
    dialog = QtGui.QFileDialog()
    dialog.setWindowTitle("Choose a file to open")
    dialog.setFileMode(QtGui.QFileDialog.ExistingFile)
    dialog.setNameFilter("Text (*.txt);; All files (*.*)")
    dialog.setViewMode(QtGui.QFileDialog.Detail)

    filename = QtCore.QStringList()

    if(dialog.exec_()):
        file_name = dialog.selectedFiles()
    plain_text = open(file_name[0]).read()
    self.Editor.setPlainText(plain_text)
    return str(file_name[0])


def window():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(20, 50, 1000, 700)
    win.setWindowTitle("GUI")

    b1 = QtWidgets.QPushButton(win)
    b1.setText("Run")
    b1.move(100, 500)
    b1.clicked.connect(run)

    b2 = QtWidgets.QPushButton(win)
    b2.setText("Test")
    b2.move(300, 500)
    b2.clicked.connect(test)

    win.show()
    sys.exit(app.exec_())

    runCommand = "python test.py"
    os.system(runCommand)

    # runCommand1 = "flex lexer.l"
    # os.system(runCommand1)
    # runCommand2 = "bison -d parser.y"
    # os.system(runCommand2)
    # runCommand3 = "g++ lex.yy.c parser.tab.c"
    # os.system(runCommand3)
    # runCommand4 = "a.exe"
    # os.system(runCommand4)
    # os.system("pause")

    # semanticErrors = open('Semantic Errors.txt', 'r')
    # symbolTable = open('Symbol Table.txt', 'r')
    # semanticErrorsString = semanticErrors.read()
    # symbolTableString = symbolTable.read()

    # symbolTableLabel = QtWidgets.QLabel(win)
    # symbolTableLabel.setText("Symbol Table")
    # symbolTableLabel.move(20, 20)

    # semanticErrorsLabel = QtWidgets.QLabel(win)
    # semanticErrorsLabel.setText("Semantic Errors")
    # semanticErrorsLabel.move(600, 20)

    # symbolTableText = QtWidgets.QLabel(win)
    # symbolTableText.setText(symbolTableString)
    # symbolTableText.move(20, 50)

    # semanticErrorsText = QtWidgets.QLabel(win)
    # semanticErrorsText.setText(semanticErrorsString)
    # semanticErrorsText.move(600, 50)
    # win.show()
    # sys.exit(app.exec_())


window()
