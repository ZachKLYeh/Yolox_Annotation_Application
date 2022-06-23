#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Zachary Yeh"
__version__ = "0.0.1"
__email__ = "zach.kl.yeh@gmail.com"

import sys
from pyqt import MainWindow, app_font, style_sheet
from PyQt5.QtWidgets import QApplication

#main function
def main():

    app = QApplication(sys.argv)
    window = MainWindow()
    app.setStyleSheet(style_sheet)
    app.instance().setFont(app_font)
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
