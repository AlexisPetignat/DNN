
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
import easyocr

class LicensePlateOCR:
    
    def __init__(self):
        self.reader = easyocr.Reader(['en'])


    def isolate_plate(self, original_image : PIL.Image | np.ndarray | tf.Tensor, bbox : tuple) -> PIL.Image:

        if isinstance(original_image, (np.ndarray, tf.Tensor)):
             original_image = tf.keras.preprocessing.image.array_to_img(original_image)
        
        x1, y1, x2, y2 = bbox
        width, height = original_image.size
        
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))
        
        return original_image.crop((x1, y1, x2, y2))

    def preprocess_plate(self, plate_image, upscale_factor=2, threshold=128):
        """
        fais les etapes :
        - upscale
        - grayscale
        - binarization
        """
        new_size = (plate_image.width * upscale_factor, plate_image.height * upscale_factor)
        processed_img = plate_image.resize(new_size, resample=Image.Resampling.LANCZOS)
        
        processed_img = ImageOps.grayscale(processed_img)

        processed_img = processed_img.point(lambda p: 255 if p > threshold else 0)
        
        return processed_img

    def perform_ocr(self, processed_image : PIL.Image) -> str:

        text = ""
        
        img_np = np.array(processed_image)
        results = self.reader.readtext(img_np, detail=0)
        text = " ".join(results)

        clean_text = ''.join(e for e in text if e.isalnum())
        return clean_text
