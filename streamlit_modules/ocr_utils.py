# OCR and receipt parsing utilities

from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import re
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_image(image):
    return pytesseract.image_to_string(image)


def extract_text_from_pdf(pdf_file):
    images = convert_from_bytes(pdf_file.read())
    text = ""
    for img in images:
        text += extract_text_from_image(img)
    return text


def parse_receipt_text(text):
    date_match = re.search(r'\d{2}/\d{2}/\d{4}', text)
    manufacture_date = datetime.now()
    if date_match:
        manufacture_date = datetime.strptime(date_match.group(), '%d/%m/%Y')
    return manufacture_date, []
