from flask import Flask, request, jsonify, render_template
from model import process_image
import logging
import os
import base64
from io import BytesIO

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)


@app.route("/", methods=["GET"])
def index():
    """Serve the single-page client UI (rendered from templates/index.html)."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_image():
    """
    Analyzes an uploaded image for faces and predicts emotions.

    Request Body:
        - Content-Type: multipart/form-data
        - Fields:
            - image (file): The image file to analyze (e.g., JPEG, PNG)

    Response Model:
        Success (200):
            {
                "results": [
                    {
                        "emotion": "string",  // Predicted emotion (e.g., "happy", "sad")
                        "face_image": "string"  // Base64-encoded JPEG image of the detected face
                    },
                    ...
                ]
            }
        Error (400):
            {
                "error": "string"  // Error message (e.g., "No image file provided")
            }
    """

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    image_path = os.path.join("uploads", image_file.filename)
    image_file.save(image_path)

    results = process_image(image_path, "cpu")

    response = []
    for face_img, emotion in results:
        buffer = BytesIO()
        face_img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        response.append({"emotion": emotion, "face_image": img_str})

    # Clean up the uploaded file after processing
    try:
        os.remove(image_path)
    except OSError as e:
        logging.warning(f"Failed to delete uploaded file {image_path}: {e}")

    return jsonify({"results": response}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
