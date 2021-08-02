import numpy as np 
from tensorflow.keras.models import model_from_json
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# from lp_char_seg import crop_characters
from lp_load_mobilenet import get_model_mobilenet, get_labels_mobilenet

# json_file = open('model/MobileNet_V2_char_recog.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# model.load_weights('model/License_char_plate_recog.h5')
# print('Model loaded successfully!')

# labels = LabelEncoder()
# labels.classes_ = np.load('model/classes.npy')
# print('Labels loaded successfully!')



def preprocess_chars_predict(image, model, labels):
    image = cv2.resize(image, (80, 80))
    image = np.stack((image, )*3, axis = -1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction


def get_plate_number(crop_characters):
    plate_number = ''
    cols = len(crop_characters)
    # fig = plt.figure(figsize=(12, 3))
    for i, char in enumerate(crop_characters):
        # plt.subplot(1, cols, i+1)
        char_label = np.array2string(preprocess_chars_predict(char, get_model_mobilenet(), get_labels_mobilenet()))
        # plt.title(char_label.strip("'[]"), fontsize=18)
        plate_number += char_label.strip("'[]")
        # plt.axis(False)
        # plt.imshow(char, cmap='gray')

    return plate_number
