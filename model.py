from typing import List, Tuple
import numpy as np
import os

import cv2
# from facenet_pytorch import MTCNN
from PIL import Image
from tensorflow import keras

import logging

logging.basicConfig(level=logging.INFO)

# Load the emotion recognition model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
emotion_model = keras.models.load_model(MODEL_PATH)

# Emotion labels (adjust based on your model's training)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


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


def predict_emotion(face_img: np.ndarray) -> str:
    """
    Predicts the emotion from a face image using the loaded Keras model.

    Args:
        face_img (np.ndarray): The face image as a numpy array (RGB format).

    Returns:
        str: The predicted emotion label.
    """
    # Preprocess the face image for the model
    # Convert to grayscale if model expects grayscale
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)

    # Resize to the expected input size (typically 48x48 for emotion recognition models)
    face_resized = cv2.resize(face_gray, (48, 48))

    # Normalize pixel values
    face_normalized = face_resized / 255.0

    # Reshape for model input (batch_size, height, width, channels)
    face_input = np.expand_dims(face_normalized, axis=0)
    face_input = np.expand_dims(face_input, axis=-1)

    # Predict emotion
    predictions = emotion_model.predict(face_input, verbose=0)
    emotion_idx = np.argmax(predictions[0])

    return EMOTION_LABELS[emotion_idx]


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

    logging.info("Starting emotion recognition for %d faces", len(facial_images))
    results = []
    for face_img in facial_images:
        emotion = predict_emotion(face_img)
        results.append((Image.fromarray(face_img), emotion))
        logging.info("Predicted emotion: %s", emotion)

    return results
