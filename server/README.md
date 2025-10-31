# Face and Emotion Recognition API

A Flask-based REST API for detecting faces in images and predicting their emotions using machine learning models.

## Features

- Face detection using MTCNN (Multi-task Cascaded Convolutional Networks)
- Emotion recognition with EmotiEffLib models
- RESTful API with JSON responses
- Base64-encoded face images in responses
- CORS support for web applications
- Automatic cleanup of uploaded files

## Installation

### Prerequisites

- Python 3.9 or higher
- UV package manager (recommended)

### Setup

1. Clone or navigate to the project directory:

   ```bash
   cd server
   ```

2. Install dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Server

Start the Flask development server:

```bash
python app.py
```

The server will run on `http://localhost:5000`.

### API Endpoint

#### POST /analyze

Analyzes an uploaded image for faces and predicts emotions.

**Request:**

- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` (file) - Image file (JPEG, PNG)

**Response (200 OK):**

```json
{
  "results": [
    {
      "emotion": "happy",
      "face_image": "base64encodedstring..."
    }
  ]
}
```

**Error Response (400 Bad Request):**

```json
{
  "error": "No image file provided"
}
```

## Project Structure

```bash
server/
├── app.py              # Flask application
├── model.py            # Face detection and emotion recognition logic
├── pyproject.toml      # Project configuration and dependencies
├── README.md           # This file
└── uploads/            # Temporary upload directory (auto-created)
```

## Dependencies

- Flask: Web framework
- EmotiEffLib: Emotion recognition library
- Facenet-Pytorch: Face detection
- OpenCV: Image processing
- Pillow: Image handling
- Flask-CORS: Cross-origin resource sharing
