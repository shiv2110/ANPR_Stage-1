import numpy as np 
import cv2
import matplotlib.pyplot as plt
from local_utils import detect_lp
from lp_load_wpodnet import get_model_wpodnet


def image_preprocessing(image, resize = False):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img/255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img


def get_license_plate(test_image, dmax=608, dmin=288):
    try:
        vehicle_img = image_preprocessing(test_image)
        ratio = float(max(vehicle_img.shape[:2])) / min(vehicle_img.shape[:2])
        side = int(ratio * dmin)
        bound_dim = min(side, dmax)
        # print(bound_dim)
        _, lp_img, _, cor = detect_lp(get_model_wpodnet(), vehicle_img, bound_dim, lp_threshold = 0.5)
        return vehicle_img, lp_img, cor
    except AssertionError as e:
        return 'Oops! Dimensions have to be tuned'


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



