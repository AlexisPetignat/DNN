
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
import easyocr

class LicensePlateOCR:
    
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def perform_ocr(self, processed_image : PIL.Image) -> str:

        text = ""
        
        img_np = np.array(processed_image)
        results = self.reader.readtext(img_np, detail=0)
        text = " ".join(results)

        clean_text = ''.join(e for e in text if e.isalnum())
        return clean_text
