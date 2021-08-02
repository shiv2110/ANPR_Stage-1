import numpy as np 
from tensorflow.keras.models import model_from_json
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from lp_load_mobilenet import get_model_mobilenet, get_labels_mobilenet



def preprocess_chars_predict(image, model, labels):
    image = cv2.resize(image, (80, 80))
    image = np.stack((image, )*3, axis = -1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction


def get_plate_number(crop_characters):
    plate_number = ''
    for i, char in enumerate(crop_characters):
        char_label = np.array2string(preprocess_chars_predict(char, get_model_mobilenet(), get_labels_mobilenet()))
        plate_number += char_label.strip("'[]")

    return plate_number
