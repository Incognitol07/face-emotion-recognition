from typing import List, Tuple
import numpy as np

from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
from PIL import Image

import logging

logging.basicConfig(level=logging.INFO)

def recognize_faces(frame: np.ndarray, device: str) -> List[np.ndarray]:
    """
    Detects faces in the given image (numpy RGB) and returns cropped face images.
    """
    def detect_face(frame: np.ndarray):
        mtcnn = MTCNN(
            keep_all=False, post_process=False, min_face_size=40, device=device
        )
        bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
        if probs is None or len(probs) == 0:
            return []
        # filter boxes by probability threshold
        valid = probs > 0.9
        if not np.any(valid):
            return []
        bounding_boxes = bounding_boxes[valid]
        return bounding_boxes

    bounding_boxes = detect_face(frame)
    logging.info("Detected %d faces", len(bounding_boxes))
    facial_images = []
    for bbox in bounding_boxes:
        box = bbox.astype(int)
        x1, y1, x2, y2 = box[0:4]
        # ensure coords are within image
        h, w = frame.shape[:2]
        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)
        facial_images.append(frame[y1:y2, x1:x2, :])
    return facial_images


def process_image(
    image_path: str, device: str = "cpu"
) -> List[Tuple[Image.Image, str]]:
    """
    Processes an input image to detect faces and predict their emotions.

    Loads the image using Pillow (no OpenCV), converts to RGB numpy array, then runs detection + emotion model.
    """
    # load image with PIL and convert to RGB
    pil_img = Image.open(image_path).convert("RGB")
    frame = np.array(pil_img)

    facial_images = recognize_faces(frame, device)

    model_name = get_model_list()[0]
    fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)

    logging.info("Starting emotion recognition for %d faces", len(facial_images))
    results = []
    for face_img in facial_images:
        emotion, _ = fer.predict_emotions(face_img, logits=True)
        results.append((Image.fromarray(face_img), emotion[0]))
        logging.info("Predicted emotion: %s", emotion[0])

    return results