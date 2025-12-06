from your_preprocess import face_detection_pipeline
from your_predict import predict_identity

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import io

from your_preprocess import face_detection_pipeline
from your_predict import predict_identity

app = Flask(__name__)

# ----------------------------
# Helper: Convert uploaded file to OpenCV format
# ----------------------------
def read_image_from_request(file):
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Convert PIL → NumPy / OpenCV (BGR)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    return img_bgr


# ----------------------------
# Endpoint: Upload and Predict
# ----------------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file found"}), 400

    file = request.files['file']
    image = read_image_from_request(file)

    # Preprocess (pipeline: detect → align → crop ROI)
    roi = face_detection_pipeline(image)

    if roi is None:
        return jsonify({"error": "Face not detected"}), 400

    # Convert ROI → grayscale → flatten as feature
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    feature_vector = roi_gray.flatten()

    # Predict identity (name)
    prediction = predict_identity(feature_vector)

    return jsonify({
        "prediction": prediction
    })


@app.route("/", methods=["GET"])
def home():
    return "Face Prediction API is running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
