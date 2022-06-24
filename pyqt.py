from document import DOCUMENT
from utils import *
import os
import cv2
from collections import Counter
from shutil import copy, rmtree
from onnxruntime import InferenceSession

#import PyQt Libraries
from PyQt5.QtWidgets import QMessageBox, QMainWindow, QFileDialog, QScrollArea
from PyQt5.QtWidgets import  QFormLayout, QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QLineEdit, QSlider, QLabel, QRadioButton, QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

#import darktheme
import qdarktheme
#set style sheet
style_sheet = qdarktheme.load_stylesheet()
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
        if self.generatethread.get_error_msg() != "":
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
        if self.calculatethread.get_error_msg() != "":
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
      self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
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
            return ""


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
            self.errormsg = ""
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