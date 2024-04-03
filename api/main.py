from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json
import boto3
import tempfile
import random
import os

AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

app = FastAPI()

#set up s3
s3_client = boto3.client('s3')

s3 = boto3.resource(
    service_name='s3',
    region_name="AWS_DEFAULT_REGION",
    aws_access_key_id="AWS_ACCESS_KEY_ID",
    aws_secret_access_key="AWS_SECRET_ACCESS_KEY"
)

BUCKET_NAME = 'modelsbucket0408'

# dictionary mapping model types to their S3 keys
model_file_keys = {
    'base_model': 'base_model.keras',
    'beta_model': 'beta_model.keras'
}

class_names_keys = {
    'base_model': 'base_model_class_names.json',
    'beta_model': 'beta_model_class_names.json'
}

def load_model_from_s3(model_type='base_model'):
    if model_type not in model_file_keys:
        raise ValueError(f"Model type '{model_type}' is not valid. Choose 'base_model' or 'beta_model'.")
    
    model_file_key = model_file_keys[model_type]
    class_names_key = class_names_keys[model_type]

    
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
        s3_client.download_fileobj(BUCKET_NAME, model_file_key, tmp_file)
        tmp_file_path = tmp_file.name

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file2:
        s3_client.download_fileobj(BUCKET_NAME, class_names_key, tmp_file2)
        tmp_file_path2 = tmp_file2.name

   

    
    # Load the model from the temporary file
    model = tf.keras.models.load_model(tmp_file_path)

    with open(tmp_file_path2, 'r') as json_file:
        class_names = json.load(json_file)

    os.remove(tmp_file_path)
    os.remove(tmp_file_path2)

    
    return model, class_names

MODEL, MODEL_CLASS_NAMES = load_model_from_s3('base_model')

BETA_MODEL,BETA_MODEL_CLASS_NAMES = load_model_from_s3('beta_model')


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get('/ping')
async def ping():
    return  'hello i am alive'


@app.post('/predict')
async def predict(
    file: UploadFile = File(...)  
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    # decide whether to use the beta model or the base model
    if random.random() < 0.1:
        model_to_use = BETA_MODEL
        class_names_to_use = BETA_MODEL_CLASS_NAMES
    else:
        model_to_use = MODEL
        class_names_to_use = MODEL_CLASS_NAMES

    
    predictions = model_to_use.predict(img_batch)
    predicted_class = class_names_to_use[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {
        'class': predicted_class,
        'confidence': float(confidence)
        }