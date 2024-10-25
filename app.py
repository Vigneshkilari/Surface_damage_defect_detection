import streamlit as st
from home import show_home_page
from predict import show_predict_page
from train import show_train_page

def main():
    page = st.sidebar.selectbox("Select a Page", ["Home", "Predict", "Train"])

    if page == "Home":
        show_home_page()
    elif page == "Predict":
        show_predict_page()
    elif page == "Train":
        show_train_page()

if __name__ == "__main__":
    main()