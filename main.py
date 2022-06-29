#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Zachary Yeh"
__version__ = "0.0.1"
__email__ = "zach.kl.yeh@gmail.com"

import sys
from pyqt import app, MainWindow

#main function
def main():

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
