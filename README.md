This is the document for this yolox annotation app

----------GETTING STARTED----------

Getting started:
This script can be wrap up into executable file via pyinstaller
The yoloxm.onnx is the onnx file generated from yolox training
You can made up the app by these two commands

1.pip install pyinstaller
2.pyinstaller main.py --onefile --add-data "yoloxm.onnx;." --windowed

The executable file reside in Yolox_Annotation_Application/dist/main.exe

----------FEATURE INTRODUCTION----------

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
  The result will be reported in a pop up window after calculation

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
