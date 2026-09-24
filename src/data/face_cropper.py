"""
Face detection and cropping module with dynamic margin expansion.
Supports .heic, .jpg, .jpeg, .png image formats.
"""

import os
import cv2
import numpy as np
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


class FaceCropper:
    def __init__(self, cascade_path=None, margin_ratio=0.20):
        self.margin_ratio = margin_ratio
        if cascade_path is None or not os.path.exists(cascade_path):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def crop_face(self, image_input, target_size=(128, 128)):
        """
        Detects primary face, applies margin ratio, crops and resizes to target_size.
        """
        if isinstance(image_input, str):
            try:
                pil_img = Image.open(image_input).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise IOError(f"Could not read image from {image_input}: {e}")
        else:
            img = image_input.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        h_img, w_img, _ = img.shape
        if len(faces) > 0:
            x, y, w, h = faces[0]
            margin = int(self.margin_ratio * w)

            start_x = max(0, x - margin)
            start_y = max(0, y - margin)
            end_x = min(w_img, x + w + margin)
            end_y = min(h_img, y + h + margin)

            cropped = img[start_y:end_y, start_x:end_x]
        else:
            cropped = img

        if target_size is not None:
            cropped = cv2.resize(cropped, target_size)

        return cropped


def detect_and_crop_face(src_path, dst_path, target_size=(128, 128), margin_ratio=0.20):
    cropper = FaceCropper(margin_ratio=margin_ratio)
    cropped = cropper.crop_face(src_path, target_size=target_size)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, cropped)
