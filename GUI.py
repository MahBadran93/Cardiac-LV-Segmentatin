# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtGui as gui




class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(639, 534)
        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(20, 50, 271, 211))
        self.graphicsView.setObjectName("graphicsView")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(30, 20, 71, 16))
        self.label.setObjectName("label")
        self.Upload_bott = QtWidgets.QPushButton(Dialog)
        self.Upload_bott.setGeometry(QtCore.QRect(20, 270, 101, 23))
        self.Upload_bott.setObjectName("Upload_bott")
        self.Upload_bott.clicked.connect(self.browseImageMAri)
        self.Segment_bott = QtWidgets.QPushButton(Dialog)
        self.Segment_bott.setGeometry(QtCore.QRect(210, 270, 80, 23))
        self.Segment_bott.setObjectName("Segment_bott")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(310, 20, 151, 16))
        self.label_2.setObjectName("label_2")
        self.image = QtWidgets.QLabel(Dialog)
        self.image.setGeometry(QtCore.QRect(320, 50, 271, 211))
        self.image.setObjectName("image")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "MRI Image"))
        self.Upload_bott.setText(_translate("Dialog", "Upload Image"))
        self.Segment_bott.setText(_translate("Dialog", "Segment"))
        self.label_2.setText(_translate("Dialog", "Segmented Image "))
        self.image.setText(_translate("Dialog", "asddsa"))
        
         
    def browseImageMAri(self):
       print('test click')
       image = QtGui.QImage(QtGui.QImageReader("../Data/TrainingImages/testImage1.png").read())
       self.image.setPixmap(QtGui.QPixmap(image))
       #open a dialog 
       fileName = QtWidgets.QFileDialog.getOpenFileName()
       print(fileName)
      

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
