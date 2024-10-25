import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
import uuid
from train_model import train_data
def show_train_page():
    col1, col2 = st.columns(2)

    # Get the images from the user for Class 0
    img_paths_0 = st.file_uploader("Upload images for Class 0", type=["jpg","jpeg","png"], accept_multiple_files=True)

    if len(img_paths_0) > 0:
        # Check if files have been uploaded
        col3, _ = st.columns(2)
        with col3:
            # Get the class from the user
            class_label = st.selectbox("Select class for Class 1", ["Class 1"])

            new_folder_name_0 = "augmented_train_data_0"

            # Check if the folder exists, otherwise create it
            if not os.path.exists(new_folder_name_0):
                os.makedirs(new_folder_name_0)

        # Loop through all uploaded images and process them for Class 0
        for i, image in enumerate(img_paths_0):
            # Load the original image
            img = Image.open(image)

            # Convert the image to RGB format
            img_data = np.array(img).astype(np.uint8)

            # Apply random flip, rotation, and brightness adjustment to each image
            folder_name = f"{new_folder_name_0}/augmented_image_{uuid.uuid4().hex}"
            for _ in range(10):
                flipped_img = np.fliplr(img_data)
                rotated_img = cv2.rotate(img_data, cv2.ROTATE_90_CLOCKWISE)
                brightened_img = cv2.convertScaleAbs(img_data, alpha=1.5, beta=10)

                # Save the augmented images
                cv2.imwrite(f"{folder_name}_flipped_{i}.jpg", flipped_img)
                cv2.imwrite(f"{folder_name}_rotated_{i}.jpg", rotated_img)
                cv2.imwrite(f"{folder_name}_brightened_{i}.jpg", brightened_img)

            st.write(f"Processing image {i+1} of {len(img_paths_0)} for Class 0")

    if len(img_paths_0) == 0:
        st.write("Please upload at least one image for Class 0.")

    # Get the images from the user for Class 1
    img_paths_1 = st.file_uploader("Upload images for Class 1", type=["jpg","jpeg","png"], accept_multiple_files=True)

    if len(img_paths_1) > 0:
        # Check if files have been uploaded
        col3, _ = st.columns(2)
        with col3:
            new_folder_name_1 = "augmented_train_data_1"

            # Check if the folder exists, otherwise create it
            if not os.path.exists(new_folder_name_1):
                os.makedirs(new_folder_name_1)

        # Loop through all uploaded images and process them for Class 1
        for i, image in enumerate(img_paths_1):
            # Load the original image
            img = Image.open(image)

            # Convert the image to RGB format
            img_data = np.array(img).astype(np.uint8)

            # Apply random flip, rotation, and brightness adjustment to each image
            folder_name = f"{new_folder_name_1}/augmented_image_{uuid.uuid4().hex}"
            for _ in range(10):
                flipped_img = np.fliplr(img_data)
                rotated_img = cv2.rotate(img_data, cv2.ROTATE_90_CLOCKWISE)
                brightened_img = cv2.convertScaleAbs(img_data, alpha=1.5, beta=10)

                # Save the augmented images
                cv2.imwrite(f"{folder_name}_flipped_{i}.jpg", flipped_img)
                cv2.imwrite(f"{folder_name}_rotated_{i}.jpg", rotated_img)
                cv2.imwrite(f"{folder_name}_brightened_{i}.jpg", brightened_img)

            st.write(f"Processing image {i+1} of {len(img_paths_1)} for Class 1")

    if len(img_paths_1) == 0:
        st.write("Please upload at least one image for Class 1.")
    train_data()