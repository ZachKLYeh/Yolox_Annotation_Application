#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Zachary Yeh"
__version__ = "0.0.1"
__email__ = "zach.kl.yeh@gmail.com"

#here is the document of this appliction
DOCUMENT = '''This is the document for this yolox annotation app
The executable file reside in "yolox/dist/main.exe"

CONTENTS:
--FEATURE INTRODUCTION
--WARNING MESSAGES
--ERROR MESSAGES

----------FEATURE INTRODUCTION:----------

There are two main features:
1.Generate xml format annotations
2.Calculate annotation instances

(1) Generate xml format annotations
1. Select the input folder and output folder:
Input folder can only contain jpg images, other files will raise an error.
The default setting is to set the output folder the same as the input folder.
You can also use the button "..." to utilize file browser

2. Select threshold:
The threshold is the confidence threshold of prediction, higher threshold will result in higher accuracy.
However, the detected object will be less. The default threshold is 0.5(maximum = 1.0)

3. Select whether to visualize:
If you select the visualize option, this will create a "visualized_images" folder in your designated output folder.
However, if there's already a "visualized_images" folder, this will raise a warning.
Making sure you are notified of overwriting these images.

4. Press 'Generate annotations' button:
This will generate the result, both annotations and visualized images.
After the generation. There will be a pop up window, indicating the annotation path and visualizing images path.

(2)Caluculate annonation instance
1. Select input folder:
The calculation only takes the input folder. Other options are not considered

2. Press "Calculate annotations" button:
This will calculate label instances in the input folder, also sub folders in the input folder
The result will be reported in a pop up window after calculation.

----------WARNING MESSAGES----------

There are two main warning messages

1.xml exist warning:
This is the case when your designated output folder already have labeled data.
There will be a warning, you can decide whether to overwrite the xml files

2.visualized image exist warning:
This is the case when your designated output folder already have visualized images.
There will be a warning, you can decide whether to overwrite the image files

----------ERROR MESSAGES----------

There are two main error messages

1.File format error:
If the input folder exist some files except for jpg and xml format, this will raise an error.

2.Invalid path error:
If your input folder or output folder path does not exist, this will raise an error
'''

#import libraries
import cv2
import numpy as np
import os
import sys
from shutil import copy, rmtree

import xml.etree.cElementTree as ET
from collections import Counter
from onnxruntime import InferenceSession

#detection classes
CLASSES = (
    "person", 
    "car", 
    "motorbike", 
    "bus", 
    "truck", 
    "bike"
)

#color map
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


#here are some global functions that would be used in the proccess

#check is the dir exist, if not, then create one
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


#preprocess, including channel transpouse, resize, and int to float transform
def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

    return padded_img, r


#nms methods
def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware

    return nms_method(boxes, scores, nms_thr, score_thr)

def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr

    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)

    if keep:
        dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1)

    return dets


#post process for visualizing images
def demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


#make xml file after model prediction
def makexml(classes, annotations, filepath, origin_img_shape, input_dir, output_dir):

    boxes, scores, cls_inds = annotations
    boxes, scores, cls_inds = boxes.astype(int), scores.astype(int), cls_inds.astype(int)
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = str(input_dir)
    ET.SubElement(annotation, 'filename').text = str(os.path.basename(filepath))
    ET.SubElement(annotation, 'path').text = str(os.path.abspath(filepath))
    
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str (origin_img_shape[1])
    ET.SubElement(size, 'height').text = str(origin_img_shape[0])
    ET.SubElement(size, 'depth').text = str(origin_img_shape[2])

    ET.SubElement(annotation, 'segmented').text = '0'

    for i in range(len(cls_inds)):
            object = ET.SubElement(annotation, 'object')
            ET.SubElement(object, 'name').text = classes[cls_inds[i]]
            ET.SubElement(object, 'pose').text = 'Unspecified'
            ET.SubElement(object, 'truncated').text = '0'
            ET.SubElement(object, 'difficult').text = '0'

            boxes[i][0] = 0 if boxes[i][0] < 0 else boxes[i][0]
            boxes[i][1] = 0 if boxes[i][1] < 0 else boxes[i][1]
            boxes[i][2] = origin_img_shape[1] if boxes[i][2] > origin_img_shape[1] else boxes[i][2]
            boxes[i][3] = origin_img_shape[0] if boxes[i][0] > origin_img_shape[0] else boxes[i][3]

            bndbox = ET.SubElement(object, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(boxes[i][0])
            ET.SubElement(bndbox, 'ymin').text = str(boxes[i][1])
            ET.SubElement(bndbox, 'xmax').text = str(boxes[i][2])
            ET.SubElement(bndbox, 'ymax').text = str(boxes[i][3])

    tree = ET.ElementTree(annotation)

    if os.path.exists(output_dir):
        mkdir(output_dir)
        xml_file_name = os.path.join(output_dir, os.path.basename(filepath).split('.')[0]+'.xml')
    else:
        xml_file_name = os.path.join(input_dir, os.path.basename(filepath).split('.')[0]+'.xml')

    tree.write(xml_file_name)


#read and xml file, accumelated the classes that it contains
def read_content(xml_file: str, classes, errormsg_record):
    finished = True
    classes = classes
    objects = []  
    list_with_all_boxes = []
    labels = []
    read_content_error_message = errormsg_record
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    filename = root.find('filename').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    for items in root.iter('object'):
        name = items.find("name").text
        
        if name not in classes:
            read_content_error_message += f'{filename} label {name} Not in classes\n'
            finished = False
            break
        
        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(items.find("bndbox/ymin").text)
        xmin = int(items.find("bndbox/xmin").text)
        ymax = int(items.find("bndbox/ymax").text)
        xmax = int(items.find("bndbox/xmax").text)
        
        area = (xmax - xmin) * (ymax - ymin)
        labels.append(name)

        if(xmin < 0 or xmin > width or ymin < 0 or ymin > height or
           xmax <= 0 or xmax > width or ymax <= 0 or ymax > height or
           area < 100):
           print(filename, xmin, ymin, xmax, ymax, area)
           read_content_error_message += f'{filename} label size[xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}, area:{area}] is invalid\n'
           finished = False
           break

        labels.append("total")

    return labels, finished, read_content_error_message


#plot bonding boxes on a image, then export
def vis(img, annonation, class_names, image_path, input_dir, output_dir, conf=0.5):

    boxes, scores, cls_ids = annonation

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    if os.path.exists(output_dir):
        mkdir(output_dir+'/visualized_images')
        output_path = os.path.join(output_dir,'visualized_images', os.path.basename(image_path))
        cv2.imwrite(output_path, img)
    else:
        mkdir(input_dir+'/visualized_images')
        output_path = os.path.join(input_dir, 'visualized_images', os.path.basename(image_path))
        cv2.imwrite(output_path, img)


#find resouce path, because the reconstructed program will not follow the directory structure
#this is mainly for searching our onnx model
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)




#import PyQt Libraries
from PyQt5.QtWidgets import QMessageBox, QMainWindow,  QApplication, QFileDialog, QScrollArea
from PyQt5.QtWidgets import  QFormLayout, QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QLineEdit, QSlider, QLabel, QRadioButton, QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
#import qdarktheme for styling
import qdarktheme

#set application font
app_font = QFont("Arial", 8)

#PyQt mainwindow
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Yolox Annotation")
        self.setFixedSize(405, 250)

        #declare some default parameters
        self.threshould = 0.5
        self.visualize = False
        self.model = "yoloxm.onnx"
        self.tutorial_string = DOCUMENT
        self.progressbar_value = 0

        #declare warnings to none
        self.warning_xml_exists = ''
        self.warning_image_exists = ''

        #The total layout
        self.layout = QVBoxLayout()

        #The form layout reside in total layout
        self.formLayout = QFormLayout()

        #input and output file selectors
        self.inputfilewidget = QWidget()
        self.filelineLayout = QHBoxLayout(self.inputfilewidget)
        self.input_dir = QLineEdit('')
        self.filelineLayout.addWidget(self.input_dir, 0)
        self.filebutton = QPushButton()
        self.filebutton.setText("···")
        self.filebutton.clicked.connect(self.inputfileselect)
        self.filelineLayout.addWidget(self.filebutton)
        self.formLayout.addRow('Input folder:', self.inputfilewidget)
        
        self.outputfilewidget = QWidget()
        self.filelineLayout = QHBoxLayout(self.outputfilewidget)
        self.output_dir = QLineEdit('Same as the input folder')
        self.filelineLayout.addWidget(self.output_dir)
        self.filebutton = QPushButton()
        self.filebutton.setText("···")
        self.filebutton.clicked.connect(self.outputfileselect)
        self.filelineLayout.addWidget(self.filebutton)
        self.formLayout.addRow('Output folder:', self.outputfilewidget)

        # HorizontalSlider for threshould selection
        self.horizontalSlider = QSlider()
        self.horizontalSlider.setOrientation(Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalSlider.valueChanged.connect(self.threshouldsliderValue)
        self.horizontalSlider.setRange(0, 10)
        self.horizontalSlider.setPageStep(1)
        self.horizontalSlider.setValue(5)
        self.horizontalSlider.setTickPosition(QSlider.TicksBelow)
        self.horizontalSlider.setTickInterval(1)
        self.formLayout.addRow('Threshold:', self.horizontalSlider)
        self.labels =  QLabel("0.0    0.1    0.2     0.3     0.4     0.5     0.6     0.7    0.8    0.9   1.0")
        self.formLayout.addRow('           ', self.labels)

        #Radiobutton to visualize selection
        self.radiobutton = QRadioButton("Visualize")
        self.radiobutton.toggled.connect(self.set_vis)
        self.formLayout.addRow(self.radiobutton)
        self.layout.addLayout(self.formLayout)

        #progress bar
        self.progressbar = QProgressBar()
        self.progressbar.setAlignment(Qt.AlignCenter)
        self.progressbar.setMaximum(100)
        self.progressbar.setValue(self.progressbar_value)
        self.layout.addWidget(self.progressbar)

        #Creat three button layout
        self.buttomwidget = QWidget()
        self.buttomLayout = QHBoxLayout(self.buttomwidget)

        #Generating button
        self.generatebutton = QPushButton()
        self.generatebutton.setText("Generate annotations")
        self.generatebutton.clicked.connect(self.generatebutton_clicked)
        self.buttomLayout.addWidget(self.generatebutton)
        
        #Calcuateing button
        self.calculatebutton = QPushButton()
        self.calculatebutton.setText("Calculate annotations")
        self.calculatebutton.clicked.connect(self.calculatebutton_clicked)
        self.buttomLayout.addWidget(self.calculatebutton)

        #Show tutorial button
        self.tutorialbutton = QPushButton()
        self.tutorialbutton.setText("Show tutorial")
        self.tutorialbutton.clicked.connect(self.tutorialbutton_clicked)
        self.buttomLayout.addWidget(self.tutorialbutton) 
        
        self.layout.addWidget(self.buttomwidget)

        #Create container
        self.container = QWidget()
        #self.container.setStyleSheet(self.stylesheet)
        self.container.setLayout(self.layout)

        # Set the central widget of the Window.
        self.setCentralWidget(self.container)

    def generatebutton_clicked(self):
        self.setWindowTitle("Generating annotations...")
        self.generatebutton.setEnabled(False)
        self.output_directory_temp = self.output_dir.text() if self.output_dir.text() != 'Same as the input folder' else self.input_dir.text()

        #check for path validity
        if self.output_directory_temp == '':
            self.output_directory_list = []
        elif os.path.exists(self.output_directory_temp) == False:
            self.output_directory_list = []
        else:
            self.output_directory_list = os.listdir(self.output_directory_temp)

        #warning detection if there is a overwrite
        for f in self.output_directory_list:
            if f == 'visualized_images':
                self.warning_image_exists = '\nThere exist visulized images in the folder, if you select the visualized option, this will overwrite these images.\n'
            if len(os.path.basename(f).split('.')) == 2:
                if os.path.basename(f).split('.')[1] == 'xml':
                    self.warning_xml_exists = '\nThere exist labeled images in the folder, this will overwrite these xml files.\n'

        #show warning or information, based on warning detection
        self.show_question(f"You are generating annotations to:\n{self.output_directory_temp}\n"+self.warning_xml_exists+self.warning_image_exists+"\nDo you want to countinue?")

        #if the proccess is approved
        if self.question_return == "OK":
            self.generatethread = GenerationThread(self.input_dir.text(), output_dir=self.output_dir.text(), model=self.model, threshould=self.threshould, visualize=self.visualize)
            #connect two important signal, finished and progressbar value
            self.generatethread.finished.connect(self.generate_finished)
            self.generatethread.generate_progressbar_value.connect(self.set_progressbar_value)
            self.generatethread.start()
        #if the proccess is denied
        else:
            self.setWindowTitle("Yolox Annotation")
            self.generatebutton.setEnabled(True)

    def generate_finished(self):
        self.generatebutton.setEnabled(True)

        #if there is an error
        if self.generatethread.get_error_msg() != "normal":
            self.set_progressbar_value(0)
            self.setWindowTitle("Yolox Annotation")
            self.show_error(self.generatethread.get_error_msg())
        #if the proccess is completed without error, show some information
        else:
            self.setWindowTitle("Annotatinos are generated!")
            if self.visualize == True:
                self.show_info(f"Annotations are generated!\nResult saved at path:\n{self.output_directory_temp}\n\nVisualized result save at path:\n{self.output_directory_temp}/visualized_images")
            if self.visualize ==False:
                self.show_info(f"Annotations are generated!\nResult saved at path:\n{self.output_directory_temp}")

    def calculatebutton_clicked(self):
        self.setWindowTitle("Calculating annotations...")
        self.calculatebutton.setEnabled(False)
        self.show_question(f"You are calculating annotations from:\n{self.input_dir.text()}\n\nDo you want to countinue?")

        #if the execution is approved
        if self.question_return == "OK":
            self.calculatethread = CalculationThread(self.input_dir.text())
            self.calculatethread.finished.connect(self.calculate_finished)
            self.calculatethread.calculate_progressbar_value.connect(self.set_progressbar_value)
            self.calculatethread.start()
        #if the execution is denied
        else:
            self.setWindowTitle("Yolox Annotation Applicaiton")
            self.calculatebutton.setEnabled(True)

    def calculate_finished(self):
        self.calculatebutton.setEnabled(True)

        #show error if there is
        if self.calculatethread.get_error_msg() != "normal":
            self.set_progressbar_value(0)
            self.setWindowTitle("Yolox Annotation")
            self.show_error(self.calculatethread.get_error_msg())
        else:
            self.setWindowTitle("Annotations are calculated!")
            self.show_report(self.calculatethread.get_result(), 'Calculation report', 'Calculation')

    def tutorialbutton_clicked(self):
        self.show_report(self.tutorial_string, 'Document', 'Document')

    def show_info(self, message):
        self.msg = QMessageBox()
        self.msg.setWindowTitle("Information")
        self.msg.setText(message)
        self.msg.setIcon(QMessageBox.Information)
        self.msg.setStandardButtons(QMessageBox.Ok)
        self.msg.exec_()

    def show_question(self, message):
        self.msg = QMessageBox()
        self.msg.setWindowTitle("Warning")
        self.msg.setText(message)

        #check if there are some warnings, if not, showinformation, else, show warnings
        if self.warning_xml_exists != '' or self.warning_image_exists != '':
            self.msg.setIcon(QMessageBox.Warning)
        else:
            self.msg.setIcon(QMessageBox.Information)
        self.msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        self.msg.buttonClicked.connect(self.question_pop_clicked)
        self.msg.exec_()

        #reset warning message after the question is answered
        self.warning_xml_exists = ''
        self.warning_image_exists = ''

    def show_report(self, string, windowtitle, mode):
        self.reportwindow = ScrollMessageBox(string, windowtitle)

        if mode == "Document":
            self.reportwindow.setStyleSheet("QScrollArea{min-width:600 px; min-height: 400px}")
        else:
            self.reportwindow.setStyleSheet("QScrollArea{min-width:400 px; min-height: 400px}")

        self.reportwindow.show()
        self.show()

    def question_pop_clicked(self, i):
        self.question_return = i.text()

    def show_error(self, error):
        self.msg = QMessageBox()
        self.msg.setWindowTitle("Error")
        self.msg.setText(error)
        self.msg.setIcon(QMessageBox.Critical)
        self.msg.setStandardButtons(QMessageBox.Ok)
        self.msg.exec_()

    def set_vis(self):
        self.visualize = True

    def threshouldsliderValue(self):
        self.threshould = self.horizontalSlider.value()/10

    def inputfileselect(self):
        self.filepath = QFileDialog.getExistingDirectory(None, "Select input directory", "C:/")
        self.input_dir.setText(self.filepath)

    def outputfileselect(self):
        self.filepath = QFileDialog.getExistingDirectory(None, "Select output directory", "C:/")
        self.output_dir.setText(self.filepath)

    def set_progressbar_value(self, value):
        self.progressbar.setValue(int(value))


#create a scrollable message box object
class ScrollMessageBox(QMessageBox):
   def __init__(self, string, windowtitle,  *args, **kwargs):
      QMessageBox.__init__(self, *args, **kwargs)

      #creat scrollable area(main object)
      self.scroll = QScrollArea(self)
      self.scroll.setWidgetResizable(True)

      #create secondary object "content"
      self.content = QWidget()
      self.scroll.setWidget(self.content)

      #attach stringlayout to content
      self.stringlayout = QVBoxLayout(self.content)

      #create label
      self.label = QLabel(string, self)
      self.label.setFont(QFont('Arial', 10))
      #attach label to stringlay to content
      self.stringlayout.addWidget(self.label)

      #attach scholl area to the full layout of Qmessage box
      #addWidget(QWidget, int r, int c, int rowspan, int columnspan)
      self.layout().addWidget(self.scroll, 0, 0, 1, self.layout().columnCount())
      self.setWindowTitle(windowtitle)



#PyQt thread: Generation annotations thread
class GenerationThread(QThread): 
    generate_progressbar_value = pyqtSignal(int)
    def __init__(self, input_dir, output_dir, model, threshould, visualize):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model = model
        self.threshould = threshould
        self.visualize = visualize
        self.errormsg = ""

    def run(self):
        #return error message after running
        self.errormsg = self.generate(self.input_dir, self.output_dir, self.model, self.threshould, self.visualize)

    def get_error_msg(self):
        return self.errormsg

    def generate(self, input_dir, output_dir, model, threshould, visualize):
        #here are some deteced errors
        self.fileformaterror = False
        self.inputdirerror = False
        self.outputdirerror = False
        self.counter = 0

        #check path validity
        if os.path.exists(input_dir):
            files = os.listdir(input_dir)
        else:
            self.inputdirerror = True
            files = []

        if output_dir != 'Same as the input folder':
            if os.path.exists(output_dir) == False:
                self.outputdirerror = True
                files = []

        #check if there's visualized images exist, if there is, overwrite it
        if os.path.exists(output_dir):
            visualized_images_dir = os.path.join(output_dir, 'visualized_images')
            if os.path.exists(visualized_images_dir):
                rmtree(visualized_images_dir)
                mkdir(visualized_images_dir)
        else:
            visualized_images_dir = os.path.join(input_dir, 'visualized_images')
            if os.path.exists(visualized_images_dir):
                rmtree(visualized_images_dir)
                mkdir(visualized_images_dir)

        #starting generation, set progress bar to 0%
        self.generate_progressbar_value.emit(0)

        for f in files:
            filepath = os.path.join(input_dir, f)
            input_shape = tuple(map(int,  ['640', '640']))
            origin_img = cv2.imread(filepath)

            #when counrtering existing xml file or visulzed images, ignore the filefromat error, keep execting
            self.ming=os.path.splitext (f)
            str = self.ming[1]
            if str == '.xml':
                self.counter += 1
                continue
            elif f == 'visualized_images':
                self.counter += 1
                continue
            elif str != '.jpg':
                self.fileformaterror = True
                break

            #starting generation
            img, ratio = preprocess(origin_img, input_shape)
            session = InferenceSession(resource_path(model))
            ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
            output = session.run(None, ort_inputs)
            predictions = demo_postprocess(output[0], input_shape, p6=False)[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
            boxes_xyxy /= ratio
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=threshould)

            if dets is not None:
                annotations = dets[:, :4], dets[:, 4], dets[:, 5]
                makexml(CLASSES, annotations, filepath, origin_img.shape, input_dir, output_dir)
                if visualize:
                    origin_img = vis(origin_img, annotations, CLASSES, filepath, input_dir, output_dir, threshould)

            self.counter += 1
            self.generate_progressbar_value.emit(int(round(self.counter*100 / len(files), 0)))

        self.generate_progressbar_value.emit(100)

        #return error status
        if self.fileformaterror == True:
            return "File format error:\nInput folder can only contain jpg images, please check your directory"
        elif self.inputdirerror == True:
            return "Input directory in invalid:\nPlease check the input directory"
        elif self.outputdirerror ==True:
            return "Output directory is invalid:\nPlease check the output directory"
        else:
            return "normal"


#PyQt thread: calculation thread
class CalculationThread(QThread): 
    calculate_progressbar_value = pyqtSignal(int)
    def __init__(self, input_dir):
        super().__init__()
        self.input_dir = input_dir
        self.result = ""
        self.errormsg = "normal"

    def run(self):
        self.result, self.errormsg = self.calculate_annotations(self.input_dir)

    def get_result(self):
        return self.result

    def get_error_msg(self):
        return self.errormsg

    def calculate_annotations(self, input_dir):
        #declare parameters
        self.input_dir = input_dir

        #classes here is different from CLASSES because we want to calculate total instances
        self.classes = ("total", "person", "car", "motorbike", "bus", "truck", "bike")
        self.analysis = []
        self.labels = []
        self.finished = True

        #Build temp directory
        self.calcuation_result_text = ""
        self.current_path = os.getcwd()
        self.temp_dir_path = os.path.join( self.current_path,"temp_dir_for_calulating_annotations_utilized_by_yolox_onnx_model_deployment")
        mkdir(self.temp_dir_path)

        #check input dir existence
        #copying files to temp directory
        if os.path.exists(self.input_dir):
            self.errormsg = "normal"
        else:
            self.errormsg = "Input directory is invalid:\nPlease check the input directory"
            self.input_dir = self.temp_dir_path

        #calculation begin
        self.calculate_progressbar_value.emit(0)

        #copy files in the main directory to temp
        for root, dirs, files in os.walk(self.input_dir):
          for f in files:
            self.fullpath = os.path.join(root, f)
            self.ming=os.path.splitext (self.fullpath)
            str=self.ming[1]
            
            if str==".xml":
                self.from_path = os.path.join(self.input_dir, self.fullpath)
                copy(self.from_path,self.temp_dir_path)

        #finished copying, then calulate temp directory
        self.calculate_progressbar_value.emit(25)

        files = os.listdir(self.temp_dir_path)

        for f in files:
            self.fullpath = os.path.join(self.temp_dir_path, f)
            self.read_labels, self.finished, self.errormsg = read_content(self.fullpath, self.classes, self.errormsg)
            self.labels.extend(self.read_labels)

        if self.finished:
            self.calcuation_result_text += 'Annotations in input folder:\n'
            self.calcuation_result_text += (f'{self.input_dir}\n')
            self.label_counter = Counter(self.labels)

            for i in self.classes:
                self.analysis.append(self.label_counter[i])
            for i in range(len(self.classes)):
                self.calcuation_result_text += f'{self.classes[i]}: '+f'{self.analysis[i]}\n'

            #reset parameters for next iteration
            self.analysis = []
            self.labels= []
            self.calcuation_result_text += '\n'
            rmtree(self.temp_dir_path)
            os.mkdir(self.temp_dir_path)

        #perform a os walk in sub dirs
        self.calculate_progressbar_value.emit(50)

        for root, dirs, files in os.walk(self.input_dir):
          for d in dirs:
            self.subdirpath = os.path.join(root,d)

            #start copying proccess in a sub dir
            for subroot, subdirs, subfiles in os.walk(self.subdirpath):
              for f in subfiles:
                self.fullpath = os.path.join(subroot, f)
                self.ming=os.path.splitext (self.fullpath)

                str=self.ming[1]
                if str==".xml":
                    self.source_path = os.path.join(self.fullpath)
                    copy(self.source_path,self.temp_dir_path)

            #calculation in the temp directory
            self.temp_dir_files = os.listdir(self.temp_dir_path)

            for f in self.temp_dir_files:
              self.readpath = os.path.join(self.temp_dir_path, f)

              if os.path.isfile(self.readpath):
                self.read_labels, self.finished, self.errormsg = read_content(self.readpath, self.classes, self.errormsg)
                self.labels.extend(self.read_labels)
              elif os.path.isdir(self.readpath):
                continue

            if self.finished:
              if Counter(self.labels)==Counter():
                pass
              else:
                self.calcuation_result_text += ('Annotations in sub folder:\n')
                self.calcuation_result_text += (f'{self.subdirpath}\n')
                self.label_counter = Counter(self.labels)

                for i in self.classes:
                    self.analysis.append(self.label_counter[i])
                for i in range(len(self.classes)):
                    self.calcuation_result_text += f'{self.classes[i]}: '+f'{self.analysis[i]}\n'

                self.calcuation_result_text += '\n'
                self.labels = []
                self.anapysis = []
                rmtree(self.temp_dir_path)
                os.mkdir(self.temp_dir_path)

            self.calculate_progressbar_value.emit(75)

        #remove temp dirctory
        rmtree(self.temp_dir_path)
        self.calculate_progressbar_value.emit(100)

        #return a string and errormessage
        return self.calcuation_result_text, self.errormsg


#main function
def main():

    app = QApplication(sys.argv)
    window = MainWindow()
    app.setStyleSheet(qdarktheme.load_stylesheet())
    app.instance().setFont(app_font)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()