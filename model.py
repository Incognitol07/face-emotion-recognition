from typing import List, Tuple
import numpy as np

import cv2
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
from PIL import Image

import logging

logging.basicConfig(level=logging.INFO)

def recognize_faces(frame: np.ndarray, device: str) -> List[np.ndarray]:
    """
    Detects faces in the given image and returns the facial images cropped from the original.

    This function reads an image from the specified path, detects faces using the MTCNN
    face detection model, and returns a list of cropped face images.

    Args:
        frame (numpy.ndarray): The image frame in which faces need to be detected.
        device (str): The device to run the MTCNN face detection model on, e.g., 'cpu' or 'cuda'.

    Returns:
        list: A list of numpy arrays, representing a cropped face image from the original image.

    Example:
        faces = recognize_faces('image.jpg', 'cuda')
        # faces contains the cropped face images detected in 'image.jpg'.
    """

    def detect_face(frame: np.ndarray):
        mtcnn = MTCNN(
            keep_all=False, post_process=False, min_face_size=40, device=device
        )
        bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
        if probs[0] is None:
            return []
        bounding_boxes = bounding_boxes[probs > 0.9]
        return bounding_boxes

    bounding_boxes = detect_face(frame)
    logging.info("Detected %d faces", len(bounding_boxes))
    facial_images = []
    for bbox in bounding_boxes:
        box = bbox.astype(int)
        x1, y1, x2, y2 = box[0:4]
        facial_images.append(frame[y1:y2, x1:x2, :])
    return facial_images


def process_image(
    image_path: str, device: str = "cpu"
) -> List[Tuple[Image.Image, str]]:
    """
    Processes an input image to detect faces and predict their emotions.

    Args:
        image_path (str): Path to the input image file.
        device (str): Device to run the models on ('cpu' or 'cuda').

    Returns:
        List[Tuple[Image.Image, str]]: List of tuples, each containing a face image (PIL Image) and its predicted emotion (str).
    """
    frame_bgr = cv2.imread(image_path)
    if frame_bgr is None:
        raise ValueError(f"Could not load image from {image_path}")
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

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