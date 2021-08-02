def OCR(blur):
    import pytesseract
    # pytesseract.pytesseract.tesseract_cmd = './Pytesseract/tesseract.exe'
    text = pytesseract.image_to_string(blur, config = '--psm 11')

    return text
