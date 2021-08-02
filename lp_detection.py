# import tensorflow as tf
import numpy as np 
from os.path import splitext
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import glob
import cv2
import matplotlib.pyplot as plt
from local_utils import detect_lp
from lp_load_wpodnet import get_model_wpodnet
# from sklearn.preprocessing import LabelEncoder



# from anpr_webapp import dmax, dmin
# from anpr_webapp import test_image


# def load_save_model(path):
#     try:
#         path_root = splitext(path)[0]
#         with open(path, 'r') as json_file:
#             model_json_file = json_file.read()
#         model = model_from_json(model_json_file, custom_objects = {})
#         model.load_weights(path_root + ".h5")
#         model.save("model/wpod-net-combined.h5")
#         print("Model has been saved successfully")
#         return model
#     except Exception as e:
#         print(e)

# arch_path = "model/wpod-net.json"
# weights_and_arch_wpodNet = load_save_model(arch_path)


# model_lpr = load_model("model/wpod-net-combined.h5")


def image_preprocessing(image, resize = False):
    # img = cv2.imread(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img/255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img


# n_images = glob.glob("plates/*.jpg")
# if len(n_images) > 0:
#     plt.figure(figsize=(16, 10))
#     img = image_preprocessing(n_images[len(n_images) - 1], True)
#     # plt.axis(False)
#     # plt.imshow(img)
# else:
#     print("No images found")


def get_license_plate(test_image, dmax=608, dmin=288):
    try:
        vehicle_img = image_preprocessing(test_image)
        # print(vehicle_img.shape[1::-1])
        # print(vehicle_img.shape[0]/vehicle_img.shape[1])
        ratio = float(max(vehicle_img.shape[:2])) / min(vehicle_img.shape[:2])
        side = int(ratio * dmin)
        bound_dim = min(side, dmax)
        # print(bound_dim)
        _, lp_img, _, cor = detect_lp(get_model_wpodnet(), vehicle_img, bound_dim, lp_threshold = 0.5)
        return vehicle_img, lp_img, cor
    except AssertionError as e:
        return 'Oops! Dimensions have to be tuned'

# test_image = n_images[len(n_images) - 1]
# vehicle, LpImg, cor = get_license_plate(test_image)

# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.axis(False)
# plt.imshow(image_preprocessing(test_image))
# plt.subplot(1,2,2)
# plt.axis(False)
# plt.imshow(LpImg[0])

def draw_box(test_image, cor, thickness = 3): 
    pts = []  
    x_cd = cor[0][0]
    y_cd = cor[0][1]
    for i in range(4):
        pts.append([int(x_cd[i]), int(y_cd[i])])
    pts = np.array(pts, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    vehicle_image = image_preprocessing(test_image)
    cv2.polylines(vehicle_image, [pts], True, (0,255,0), thickness)
    return vehicle_image

# vehicle_image = draw_box(test_image, cor)
# plt.figure(figsize=(8,8))
# plt.axis(False)
# plt.imshow(vehicle_image)



