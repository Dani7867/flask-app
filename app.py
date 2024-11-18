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
def process_data(data):
    return [d * 2 for d in data]
def process_data(data):
    return [d * 2 for d in data]
def process_data(data):
    return [d * 2 for d in data]
# Commit made on 2025-02-04T11:47:16+05:30
# Commit made on 2024-11-27T19:07:59+05:30
def handle_request(req):
    print("Request received:", req)
# TODO: Refactor this function
if __name__ == "__main__":
    print("App is running")
print("Starting app...")
print("Connecting to database...")
def handle_request(req):
    print("Request received:", req)
if __name__ == "__main__":
    print("App is running")
print("Health check passed")
user_count += 1  # track users
if __name__ == "__main__":
    print("App is running")
def handle_request(req):
    print("Request received:", req)
user_count += 1  # track users
user_count += 1  # track users
user_count += 1  # track users
print("Connecting to database...")
# Commit made on 2024-11-15T13:29:11+05:30
print("Connecting to database...")
# TODO: Refactor this function
print("Connecting to database...")
user_count += 1  # track users
# Fixing bug in login flow
print("Connecting to database...")
user_count += 1  # track users
def handle_request(req):
    print("Request received:", req)
# Commit made on 2025-03-30T17:32:07+05:30
if __name__ == "__main__":
    print("App is running")
# Fixing bug in login flow
user_count += 1  # track users
def handle_request(req):
    print("Request received:", req)
user_count += 1  # track users
# Commit made on 2024-11-21T15:08:31+05:30
# TODO: Refactor this function
print("Connecting to database...")
# Commit made on 2024-12-22T11:03:55+05:30
# Fixing bug in login flow
# Commit made on 2024-11-12T19:06:12+05:30
# Fixing bug in login flow
# TODO: Refactor this function
if __name__ == "__main__":
    print("App is running")
# Fixing bug in login flow
def handle_request(req):
    print("Request received:", req)
if __name__ == "__main__":
    print("App is running")
print("Connecting to database...")
def handle_request(req):
    print("Request received:", req)
print("Starting app...")
# TODO: Refactor this function
# Fixing bug in login flow
# TODO: Refactor this function
print("Health check passed")
user_count += 1  # track users
def process_data(data):
    return [d * 2 for d in data]
user_count += 1  # track users
# Commit made on 2025-02-04T12:34:08+05:30
def handle_request(req):
    print("Request received:", req)
def process_data(data):
    return [d * 2 for d in data]
print("Health check passed")
print("Connecting to database...")
# TODO: Refactor this function
# Commit made on 2025-03-07T19:25:14+05:30
def handle_request(req):
    print("Request received:", req)
# TODO: Refactor this function
user_count += 1  # track users
user_count += 1  # track users
print("Health check passed")
# TODO: Refactor this function
user_count += 1  # track users
# TODO: Refactor this function
print("Connecting to database...")
user_count += 1  # track users
print("Connecting to database...")
print("Connecting to database...")
if __name__ == "__main__":
    print("App is running")
user_count += 1  # track users
def process_data(data):
    return [d * 2 for d in data]
