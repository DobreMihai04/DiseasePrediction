# Plant Disease Prediction

## Introduction
This application utilizes machine learning to identify plant diseases through image analysis, offering a quick and accessible preliminary diagnostic tool for anyone involved in plant care.
The model is based on the MobileNetV2 architecture, fine-tuned to plant disease prediction.

## Live Demo
You can access the live application [here](http://www.dobremihai.com/).

## Technologies
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **ML Framework**: TensorFlow
- **Cloud Hosting**: AWS EC2
- **Containerization**: Docker

## Runing it
Ensure Docker is installed on your machine and the repo is cloned, then:


```bash
cd DiseasePrediction

docker-compose up -d
```
*predictions require AWS credentials since the models are stored on S3
