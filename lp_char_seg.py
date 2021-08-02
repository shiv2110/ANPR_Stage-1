import cv2

def plate_preprocessing(LpImg):
    if len(LpImg):
        plate = cv2.convertScaleAbs(LpImg[0], alpha = (255.0))
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dilated = cv2.morphologyEx(binary, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    return plate, binary, dilated, blur, gray



def sort_contours(cnts, binary, plate, reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key = lambda b: b[1][i], reverse = reverse))
    return cnts

def find_contours(binary):
    cont, _  = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def store_chars(hp, digit_w, digit_h, ratio_up, plate, dilated, cont, binary):
    test_roi = plate.copy()
    crop_characters = []
    crop_characters.clear()
    for c in sort_contours(cont, binary, plate):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 1<=ratio<=ratio_up:
            if h/plate.shape[0]>=hp: 
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (255, 0, 255), 2)
                curr_num = dilated[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

    return test_roi, crop_characters

