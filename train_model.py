import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
import numpy as np

def train_data():
    # Define constants
    CATEGORIES = ['Defect', 'Normal']
    IMG_SIZE = 60
    DIRECTORY = r'C:/_Vignesh_N/Vit_Project/spurgear/Dataset'

    # Data preparation
    data = []
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            try:
                arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if arr is None:
                    raise Exception(f"Failed to read image: {img_path}")
                new_arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE))
                data.append([new_arr, CATEGORIES.index(category)])
            except Exception as e:
                print(f"Error processing image: {img_path}")
                print(e)

    # Shuffle and split the data
    np.random.shuffle(data)
    X = np.array([entry[0] for entry in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    y = np.array([entry[1] for entry in data])
    y = tf.keras.utils.to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the CNN model
    def create_cnn_model():
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(len(CATEGORIES), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # Define the ResNet model
    def create_resnet_model():
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        model = Sequential([
            base_model,
            Flatten(),
            Dense(len(CATEGORIES), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # Create and train models
    cnn_model = create_cnn_model()
    history_cnn = cnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    resnet_model = create_resnet_model()
    X_train_resnet = np.concatenate([X_train, X_train, X_train], axis=-1)  # Convert grayscale to 3 channels
    X_test_resnet = np.concatenate([X_test, X_test, X_test], axis=-1)
    history_resnet = resnet_model.fit(X_train_resnet, y_train, epochs=10, validation_data=(X_test_resnet, y_test))
    # Predictions
    cnn_preds = cnn_model.predict(X_test)
    resnet_preds = resnet_model.predict(X_test_resnet)
    cnn_model.save('cnn_model.keras') 
    resnet_model.save('resnet_model.keras')
train_data()