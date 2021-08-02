from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import numpy as np



json_file = open('model/MobileNet_V2_char_recog_2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model/License_char_plate_recog_2.h5')
print('Model loaded successfully!')

labels = LabelEncoder()
labels.classes_ = np.load('model/classes_2.npy')
print('Labels loaded successfully!')


def get_model_mobilenet():
    return model

def get_labels_mobilenet():
    return labels
