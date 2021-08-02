# Vehicle Number Plate Recognition

It involves three steps, namely :

1. Plate Detection
2. Character Segmentation 
3. Character Recognition

- ### Training the model with 36 classes 
    - The trained weights will be used later on for recognizing the number plate characters

    - [Get the link to colab notebook 👉 (https://colab.research.google.com/drive/1KPih7z3xRJuOimmiLbohAs3vHHJy7oA4)]

    - MobileNetV2 has been used. '''include_top=False''', this way the fully connected layer will not be included and it can be custom added by us

    - Keras-tuner 👉 (https://www.tensorflow.org/tutorials/keras/keras_tuner) has been used to select best number of units and an optimum learning rate along with best number of epochs

    - Model weights, class labels and architecture has been saved in the '''models''' directory

- ### License Plate Detection
    - WPOD-NET (Warped Planar Object Detection Network 👉 https://openaccess.thecvf.com/content_ECCV_2018/papers/Sergio_Silva_License_Plate_Detection_ECCV_2018_paper.pdf?ref=hackernoon.com) can detect and extract license plates from 10 different countries and can also detect multiple plates in one image. 

- ### License Plate Character Segmentation
    - This is done by using contouring, drawing BB (boundary boxes) and cropping the characters along the boundary boxes

- ### License Plate Character Recognition
    - Each cropped character of the number plate is classified and the number of the plate is obtainned.

    - A two step process for obtaining the number in text format is to use python's OCR (Optical Character Recognition) library called '''pytesseract'''




[Refer to openCV documentation 👉 (https://docs.opencv.org/master/d6/d00/tutorial_py_root.html) ]