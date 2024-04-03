# Plant Disease Prediction

## Introduction
This application utilizes machine learning to identify plant diseases, offering a quick and accessible preliminary diagnostic tool for anyone involved in plant care.
The model is based on the MobileNetV2 architecture, fine-tuned to plant disease prediction.

## Live Demo
You can access the live application [here](http://www.dobremihai.com/).

## Technologies
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **ML Framework**: TensorFlow
- **Cloud Hosting**: AWS EC2
- **Containerization**: Docker

## Running it
*the backend requires AWS credentials since the models are stored on S3

### Setup:
1.  Clone this repository
2.  Create a python3 virtual enviroment

     ```   
    python3 -m venv .env
    ```
4.  Activate the virtual enviroment
    ```   
    source .env/bin/activate
    ```
5.  Install dependencies
    ```   
    pip3 install -r api/requirements.txt
    
    pip3 install -r streamlit/requirements.txt
    ```
### Run: 
1. Run server
    ```
    cd api
    
    uvicorn main:app --reload --host localhost --port 8000
    ```
    
2. Run streamlit

    ```
    cd streamlit
    
    streamlit run streamlit_code.py
    ```
