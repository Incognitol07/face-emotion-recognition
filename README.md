# Face & Emotion Recognition — Flask API + HTML Client

A project that detects faces in images and predicts emotions for each detected face.

This repo contains a Flask-based backend that accepts image uploads and returns predicted
emotions along with base64-encoded face thumbnails, and a client (`client/index.html`)
that captures an image from the user's camera (or uploaded image) and sends it to the API.

---

## Quick summary

- Backend: `server/app.py` — Flask app exposing POST /analyze
- Model logic: `server/model.py` — face detection (MTCNN) + emotion recognition (EmotiEffLib)
- Frontend: `client/index.html` — simple page that captures video, posts image to the API, and shows results

## Features

- Detect faces using MTCNN (facenet-pytorch)
- Predict emotions using EmotiEffLib (ONNX engine)
- API returns JSON with predicted emotion and base64-encoded JPEG of each detected face
- CORS enabled so a web client can call the API from a browser

## Project structure

```
face-emotion-recognition/
├─ client/
│  └─ index.html        # Minimal UI that captures video and posts to /analyze
├─ server/
│  ├─ app.py            # Flask application and /analyze endpoint
│  ├─ model.py          # Face detection & emotion recognition logic
│  ├─ requirements.txt  # Python dependencies used by the server
│  └─ README.md         # (server-specific readme)
└─ README.md            
```

## Setup (Windows / cmd.exe)

1. Create and activate a virtual environment (recommended):

   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install server dependencies:

   ```cmd
   pip install -r server\requirements.txt
   ```

3. Start the Flask server:

   ```cmd
   python server\app.py
   ```

4. Open `client/index.html` in your browser (or serve the `client` folder with a static server) and use the UI to capture an image and send it to the server.

By default the Flask app listens on port 5000 (<http://localhost:5000>).

## API — POST /analyze

Endpoint: `http://<host>:5000/analyze`

Request:

- Method: POST
- Content-Type: multipart/form-data
- Field: `image` — image file (JPEG/PNG)

Response (200):

```json
{
  "results": [
    {
      "emotion": "happy",
      "face_image": "<base64-jpeg-string>"
    }
  ]
}
```

Error (400):

```json
{ "error": "No image file provided" }
```

Notes from the implementation:

- Uploaded files are temporarily saved to `uploads/` (the server creates that directory automatically) and deleted after processing.
- The model code in `server/model.py` uses OpenCV to read images and MTCNN (facenet-pytorch) to detect faces.
- Emotion inference uses `EmotiEffLibRecognizer` and returns a single predicted emotion per face.

The server will respond with a JSON payload as shown above; `face_image` values are base64-encoded JPEGs and can be displayed in a browser by prefixing with `data:image/jpeg;base64,`.

## Author

Abraham Adelodun — CSC334

Student ID: 23CG034020
