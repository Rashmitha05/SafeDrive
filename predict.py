import tensorflow as tf
import numpy as np
import cv2
import pyttsx3
import time

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

# Load the saved CNN model
model = tf.keras.models.load_model('vgg_model.keras')

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set properties for voice (you can customize the voice and rate if needed)
engine.setProperty('rate', 150)  # Speed of speech

# Dictionary to track whether alerts have been spoken for each action
alert_spoken = {1: False, 2: False, 3: False, 4: False, 5: False,
                6: False, 7: False, 8: False, 9: False}

# Function to speak a message
def speak_message(message):
    print(f"Speaking: {message}")  # Debugging statement
    engine.say(message)
    engine.runAndWait()  # Ensure the engine waits until the message is spoken

# Image preprocessing function for video frames and images
def preprocess_image(frame):
    IMG_SIZE = (256, 256)

    # Resize the image to the required input size for the model
    image = cv2.resize(frame, IMG_SIZE)
    
    # Convert the image from BGR to RGB (as OpenCV loads in BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize the image by rescaling pixel values to the range [0, 1]
    image = image / 255.0
    
    # Expand the dimensions to match the input shape (1, 256, 256, 3)
    image = np.expand_dims(image, axis=0)
    
    return image

# Prediction function
def predict_action(frame):
    # Preprocess the frame
    processed_image = preprocess_image(frame)

    # Perform prediction
    predictions = model.predict(processed_image)

    # Get the predicted class index with the highest probability
    predicted_class_index = np.argmax(predictions)

    # Get the corresponding class label and action name
    predicted_action = class_labels[predicted_class_index]

    return predicted_action, predicted_class_index

# Function to process video input
def process_video(video_path):
    global alert_spoken  # Use the global variable

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("End of video or error in reading frame.")
                break

            # Predict driver action based on the current frame
            predicted_action, predicted_class_index = predict_action(frame)

            # Display the resulting frame with prediction
            cv2.putText(frame, f"Action: {predicted_action}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check if the predicted action is a distracted action (classes 1-9)
            if predicted_class_index in alert_spoken and not alert_spoken[predicted_class_index]:
                # Define alert messages
                alert_message_en = f"Concentrate on driving rather than on {predicted_action.lower()}."
                alert_message_te = f"డ్రైవింగ్‌పై దృష్టి పెట్టకుండా {predicted_action.lower()} పై దృష్టి పెట్టండి."

                # Speak the alert messages
                speak_message(alert_message_en)
                speak_message(alert_message_te)

                # Mark the alert as spoken for this action
                alert_spoken[predicted_class_index] = True

            # Reset the alert flag if the driver is safe or if any alert should be reset
            if predicted_class_index == 0:  # Safe Driving
                # Reset all alerts
                alert_spoken = {1: False, 2: False, 3: False, 4: False, 5: False,
                                6: False, 7: False, 8: False, 9: False}

            # Show the frame
            cv2.imshow('Driver Distraction Detection', frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Function to process image input
def process_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Check if image is loaded successfully
    if image is None:
        print("Error: Image not loaded properly!")
        return None

    # Predict driver action based on the input image
    predicted_action, _ = predict_action(image)

    if predicted_action is not None:
        print(f"Predicted Action: {predicted_action}")
    else:
        print("Prediction failed.")

# Main function
if __name__ == "__main__":
    choice = input("Choose input type (image/video): ").strip().lower()

    if choice == 'image':
        image_path = input("Enter the path of the image: ")
        process_image(image_path)

    elif choice == 'video':
        video_path = input("Enter the path of the video (.mp4): ")
        process_video(video_path)

    else:
        print("Invalid choice. Please select either 'image' or 'video'.")
