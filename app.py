from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Load pre-trained model
model = load_model('saved_model/best_model.keras')

# Emotion labels
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

@app.route('/')
def home():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_emotion():
    """Endpoint to predict emotion from an uploaded image."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file:
        return jsonify({"error": "Invalid file"}), 400

    try:
        # Read the image
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        
        # Preprocess the image
        img_resized = cv2.resize(img, (48, 48))
        img_resized = img_resized / 255.0  # Normalize pixel values
        img_resized = np.expand_dims(img_resized, axis=-1)  # Add channel dimension
        img_resized = np.expand_dims(img_resized, axis=0)   # Add batch dimension
        
        # Make prediction
        prediction = model.predict(img_resized)
        maxindex = int(np.argmax(prediction))
        emotion = emotion_dict[maxindex]
        
        return jsonify({"emotion": emotion})
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

if __name__ == '__main__':
    # Ensure 'best_model.keras' exists before starting the server
    model_path = 'saved_model/best_model.keras'
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Please ensure the file is in the correct location.")
        exit(1)
    app.run(debug=True)
print("Health check passed")
if __name__ == "__main__":
    print("App is running")
user_count += 1  # track users
# TODO: Refactor this function
# TODO: Refactor this function
print("Connecting to database...")
print("Starting app...")
def process_data(data):
    return [d * 2 for d in data]
# TODO: Refactor this function
# Commit made on 2025-04-02T13:11:13+05:30
print("Health check passed")
if __name__ == "__main__":
    print("App is running")
def handle_request(req):
    print("Request received:", req)
def handle_request(req):
    print("Request received:", req)
print("Starting app...")
if __name__ == "__main__":
    print("App is running")
