import streamlit as st

def show_home_page():
    st.title("AI Based Defect Detection and Quality Assurance")

    # Create a container to hold our content
    with st.container():
        st.header("Welcome to Vision Squad")

        # Add an explanation about the project and problem statement
        st.write(
            """
            In today's Industry 4.0 era, manufacturers face increasing pressure to improve product quality and reduce defects while maintaining high production volumes.
            Manual inspection processes are time-consuming, costly, and prone to human error, leading to significant economic losses.

            Vision Squard aims to provide a solution for this problem by leveraging AI-based defect detection and quality assurance.
            """
        )

        # Proposed Solution
        st.header("Proposed Solution")

       
        st.write(
                """
                Our proposed solution involves developing an AI-powered web-based platform that integrates machine learning (ML) and deep learning (DL) models to automate defect detection and quality
assurance.
                """
            )

        # Tech Stack
        st.header("Technology Stack")

        
        st.write(
                """
                We will be using the following technologies for our project:
                * **Python**: As the primary programming language for developing our ML models and web application.
                * **TensorFlow**: For building and training our DL models.
                * **Keras**: A high-level neural networks API to simplify the development of complex ML models.
                * **Scikit-learn**: A library for ML algorithms that provides a wide range of tools for data preprocessing, feature selection, and model evaluation.
                """
            )

        # Models Used
        st.header("Models Used")

       
        st.write(
                """
                We will be using the following models for our project:
                * **Convolutional Neural Networks (CNNs)**: For image classification and defect detection tasks.
                * **Recurrent Neural Networks (RNNs)**: For processing sequential data such as production lines.
                """
            )

        # Image
        st.header("System Architecture")
        st.image("C:/_Vignesh_N/Vit_Project/final/assets/System_architechture.png", caption="System Architecture")

