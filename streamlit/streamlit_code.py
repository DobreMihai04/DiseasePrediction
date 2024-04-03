import streamlit as st
import requests
from PIL import Image
import os
from io import BytesIO
import base64

url = os.environ.get("API_URL", "http://localhost:8000/predict")

example_image_path_1 = "example_image_2.JPG"
example_image_path_2 = "example_image_1.JPG"

# function to handle predictions
def predict_and_display(image_bytes):
    with result_placeholder.container():
         st.markdown(
            f"""
            <div style='text-align: center;'>
                <img src="data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}" width="300" />
            </div>
            """,
            unsafe_allow_html=True,
        )
         
         st.markdown("<br>", unsafe_allow_html=True)
        # Display a message while the request is being processed
         with st.spinner('Processing... Please wait.'):
            # Send the request to FastAPI server
            files = {"file": image_bytes}
            try:
                response = requests.post(url, files=files)
                if response.status_code == 200:
                    result = response.json()
                    disease = result["class"]
                    confidence = result["confidence"] * 100
                    st.success(f"{disease} with confidence {confidence:.2f}%")
                else:
                    st.error("Error in prediction. Please try again.")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the server: {e}")

st.set_page_config(
    page_title="Plant Disease Prediction",
    initial_sidebar_state="expanded",
)

# apply custom theme
st.markdown("""
    <style>
    .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    .centered-column-content {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    }
    </style>
    """, unsafe_allow_html=True)

# title
st.markdown("""
    <h1 style='text-align: center; margin-top: 25px;'>Plant Disease Prediction</h1>
    """, unsafe_allow_html=True)
# sidebar
with st.sidebar:
    st.info("Instructions")
    st.write("Upload a photo of a plant, and the model will predict it's disease.")

# file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# placeholder for the prediction result
result_placeholder = st.empty()

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    predict_and_display(image_bytes)

# example images section
st.markdown("""
    <h2 style='text-align: center; margin-bottom: 25px;'>Try with Example Images</h2>
    """, unsafe_allow_html=True)
example_col1, example_col2 = st.columns(2)

example_col1, example_col2 = st.columns(2)

with example_col1:
    left_padding_col1, content_col1, right_padding_col1 = st.columns([1, 10, 1])
    with content_col1:
        if st.button("Predict Example Disease 1", use_container_width=True):
            example_image_1_bytes = open(example_image_path_1, "rb").read()
            predict_and_display(example_image_1_bytes)
        st.image(example_image_path_1, width=275) 
        
with example_col2:
    left_padding_col2, content_col2, right_padding_col2 = st.columns([1, 10, 1])
    with content_col2:
        if st.button("Predict Example Disease 2", use_container_width=True):
            example_image_2_bytes = open(example_image_path_2, "rb").read()
            predict_and_display(example_image_2_bytes)
        st.image(example_image_path_2, width=275)  # Image width reduced for centering
        
    
st.write("""
### About the Model
The deep learning model behind this application has been trained on a comprehensive dataset of common crop leaves with various disease patterns. It can currently identify multiple diseases, including but not limited to:

- Late Blight
- Early Blight
- Blackleg
- Mosaic Virus
- Black Rot

The model uses the common arhitecture of MobileNetV2 trained on ImageNet, which was fine tuned to correctly categorise plant diseases.If you are a farmer or researcher, you can use this tool as an early indicator. However, for more accurate diagnosis, please consider laboratory testing.

### Confidence Score
The confidence score reflects the model's certainty about its prediction. A higher score indicates greater confidence. If the score is low, you may want to consult a specialist or consider other factors.

### Feedback
Your feedback is valuable. If the prediction matches the actual condition of your plant, please let us know. If it doesn't, we're also interested in hearing from you. Accurate user reports will help us to improve our model.

You can send your feedback to [mihaidobre0408@gmail.com](mailto:mihaidobre0408@gmail.com).
""")