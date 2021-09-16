import os                       # for working with files
import numpy as np              # for numerical computationss
import pandas as pd             # for working with dataframes
import json
import io
from PIL import Image
from pydantic import BaseModel
import torch                    # Pytorch module 
import torchvision.transforms as transforms   # for transforming images into tensors 
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from colabcode import ColabCode
import pickle
from fastapi.middleware.cors import CORSMiddleware
from utils.model import ResNet9
import uvicorn

with open('crop_dict.json') as json_file:
    data1 = json.load(json_file)

app = FastAPI()
#api = Api(app)


#model = pickle.load(open('cropped_model.pkl','rb'))


disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']



disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load('state_dict.pth', map_location=torch.device('cpu')))

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    yb = model(img)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction



transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

class Crop(BaseModel):
    N: float 
    K: float 
    P: float 
    temperature: float 
    humidity: float 
    ph: float 
    rainfall: float 
    class Config:
        schema_extra = {
            "example": {
                "N": 93, 
                "K": 56, 
                "P": 42,
                "temperature": 23.857240,
                "humidity": 82.225730,
                "ph": 7.382763,
                "rainfall": 195.094831
            }
        }

class Prediction(BaseModel):
  filename: str
  contenttype: str
  prediction: str
# Define the /prediction route

@app.on_event("startup")
def load_model():
    global model
    model = pickle.load(open("cropped_model.pkl", "rb"))

@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}


@app.post('/cropprediction')
def get_crop_prediction(data: Crop):
    received = data.dict()
    N = received['N']
    K = received['K']
    P = received['P']
    temperature = received['temperature']
    humidity = received['humidity']
    ph = received['ph']
    rainfall = received['rainfall']
    pred_name = model.predict([[N, K, P,
                                temperature, humidity, ph, rainfall]]).tolist()[0]
    #print(str(pred_name))
    return {'prediction': data1[str(pred_name)]}

@app.post('/cropdiseaseprediction/', response_model=Prediction)
async def prediction_route(file: UploadFile = File(...)):
    contents = await file.read()
    print(file.filename)
    image = Image.open(io.BytesIO(contents))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    prediction = predict_image(img_u,disease_model)
    return {
      'filename': file.filename,
      'contenttype': file.content_type,
      'prediction': prediction
    }

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if __name__ == "__main__": 
    uvicorn.run(app)