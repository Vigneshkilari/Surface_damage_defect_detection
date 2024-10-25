import streamlit as st
import cv2
import numpy as np
import time
import tensorflow as tf

# Load pre-trained models
loaded_cnn = tf.keras.models.load_model('C:/_Vignesh_N/Vit_Project/final/models/cnn_model.keras')
loaded_resnet = tf.keras.models.load_model('C:/_Vignesh_N/Vit_Project/final/models/resnet_model.keras')

# Define constants
CATEGORIES = ['Defect', 'Normal']
IMG_SIZE = 60

def show_predict_page():
    st.title("Webcam Live Feed with Predictions")

    # Create a placeholder for the webcam feed
    frame_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break

        # Make a prediction on the current frame
        predicted_class = predict_frame(frame)

        # Convert the frame from BGR to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw the predicted class onto the frame
        cv2.putText(frame, predicted_class, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the current frame in the placeholder
        frame_placeholder.image(frame, channels="RGB", use_column_width=True)

        # Add a small delay to control the frame rate
        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()

def predict_frame(frame):
    # Preprocess the frame for prediction
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.resize(gray_frame, (IMG_SIZE, IMG_SIZE))
    gray_frame = np.array(gray_frame) / 255.0
    gray_frame = gray_frame.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Prepare the frame for the ResNet model (3 channels)
    frame_resnet = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resnet = cv2.resize(frame_resnet, (IMG_SIZE, IMG_SIZE))
    frame_resnet = np.array(frame_resnet) / 255.0
    frame_resnet = frame_resnet.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    # Make predictions with loaded models
    cnn_preds = loaded_cnn.predict(gray_frame)
    resnet_preds = loaded_resnet.predict(frame_resnet)

    # Combine predictions
    #final_preds = np.argmin(cnn_preds + resnet_preds, axis=1)
    final_preds = np.argmin(resnet_preds, axis=1)

    return CATEGORIES[final_preds[0]]  # Return the predicted class as a string
