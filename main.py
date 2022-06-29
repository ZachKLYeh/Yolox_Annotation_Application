#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Zachary Yeh"
__version__ = "0.0.1"
__email__ = "zach.kl.yeh@gmail.com"

from pyqt import app, MainWindow

#main function
def main():

    window = MainWindow()
    window.show()
    app.exec()

if __name__ == '__main__':
    main()
