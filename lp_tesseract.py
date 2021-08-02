def OCR(blur):
    import pytesseract
    text = pytesseract.image_to_string(blur, config = '--psm 11')

    return text
