"""
Real-time Face Identification for Edge Deployment on Raspberry Pi 4.
Runs camera capture, face detection/cropping, TFLite inference, and bounding-box overlay.
"""

import time
import cv2
import numpy as np


class RaspberryPiFaceRecognizer:
    def __init__(self, tflite_model_path, class_names, input_size=(128, 128)):
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite

        self.interpreter = tflite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.class_names = class_names
        self.input_size = input_size

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def predict_face(self, face_bgr):
        resized = cv2.resize(face_bgr, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        # MobileNetV2 preprocessing: [-1, 1]
        norm = (rgb / 127.5) - 1.0
        input_tensor = np.expand_dims(norm, axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        pred_idx = int(np.argmax(output))
        confidence = float(output[pred_idx])
        return self.class_names[pred_idx], confidence

    def run_live_stream(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        print("Starting real-time edge face identification. Press 'q' to quit.")

        prev_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                margin = int(0.2 * w)
                sx = max(0, x - margin)
                sy = max(0, y - margin)
                ex = min(frame.shape[1], x + w + margin)
                ey = min(frame.shape[0], y + h + margin)

                face_crop = frame[sy:ey, sx:ex]
                if face_crop.size == 0:
                    continue

                label, conf = self.predict_face(face_crop)
                text = f"{label} ({conf * 100:.1f}%)"

                cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)
                cv2.putText(frame, text, (sx, max(sy - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Compute and show live FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Raspberry Pi Face Identification", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
