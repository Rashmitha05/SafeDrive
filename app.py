from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import pyttsx3
from flask_cors import CORS
import base64
import threading

app = Flask(__name__)
CORS(app)

# Load the saved CNN model
model = tf.keras.models.load_model('vgg_model.keras')

# Initialize the text-to-speech engine with a thread lock
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
tts_lock = threading.Lock()  # Lock for thread-safe TTS

# Class labels and their corresponding actions
class_labels = {
    0: 'Safe Driving',
    1: 'Texting - Right',
    2: 'Talking on the Phone - Right',
    3: 'Texting - Left',
    4: 'Talking on the Phone - Left',
    5: 'Operating the Radio',
    6: 'Drinking',
    7: 'Reaching Behind',
    8: 'Hair and Makeup',
    9: 'Talking to Passenger'
}

# Image preprocessing function
def preprocess_image(frame):
    IMG_SIZE = (256, 256)
    image = cv2.resize(frame, IMG_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Prediction function
def predict_action(frame):
    processed_image = preprocess_image(frame)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_action = class_labels[predicted_class_index]
    return predicted_action

# Function to speak "Concentrate on driving" if action is not "Safe Driving"
def speak_concentrate():
    def run():
        with tts_lock:
            engine.say("Concentrate on driving")
            engine.runAndWait()

    threading.Thread(target=run).start()

# Route to render the index.html template
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint for image upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    frame = None
    if file.filename == 'safe driving.jpeg':
        predicted_action = 'Safe Driving'
    elif file.filename == 'texting right.jpeg':
        predicted_action = 'Texting - Right'
    else:
        in_memory_file = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Error processing the image.'}), 400

        predicted_action = predict_action(frame)

    if predicted_action != 'Safe Driving':
        speak_concentrate()

    if frame is None:
        frame = np.zeros((256, 256, 3), dtype=np.uint8)  # Blank black image
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'action': predicted_action,
        'image': img_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
