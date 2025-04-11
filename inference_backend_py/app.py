import sys; sys.path.append('.')
import torch
import torchvision
import os
import time

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
import json

from typing import Annotated
from pydantic import BaseModel

import base64

from PIL import Image
import io

from train import CLASS_MAP, CLASS_MAP_INV, get_device, get_test_transform

logging=False

def load_model(model_path: str, device: torch.device = None):

    print(f"Loading model from {model_path} on device {device}")
    
    if not model_path.endswith('.pt'):
        raise ValueError("Model path must end with .pt")

    model = torch.load(model_path, weights_only=False, map_location=device)
    print("Model loaded successfully")

    return model


def get_image_tensor_from_base64_string(base64_string: str):  
    png_data = base64.b64decode(base64_string.replace(' ', '+')) # this took a disgusting amount of time to figure out

    image = Image.open(io.BytesIO(png_data))

    image = image.convert('RGB')

    image = torch.tensor(image)

    image = app.state.image_transform(image)

    return image


def get_image_from_path(image_path: str, device=None):
    img_as_tensor = torchvision.io.read_image(image_path)

    img_as_tensor = torch.tensor(img_as_tensor)

    img_as_tensor = app.state.image_transform(img_as_tensor)

    img_as_tensor = img_as_tensor.to(device)

    return img_as_tensor


# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    with open('inference_backend_py/app_parameters.json', 'r') as f:
        args = json.load(f)

    print("Getting device...")
    device = get_device(use_gpu=args['use_gpu'])
    print(f"Device: {device}")

    app.state.device = device
    
    model = load_model(args['model_path'], device=app.state.device)

    app.state.model = model

    image_transform = get_test_transform()

    app.state.image_transform = image_transform

    app.state.model_name = args['model_name']

    yield

    # put any cleanup code here (stuff to do right before the app shuts down)

app = FastAPI(lifespan=lifespan)

# this makes it so that the frontend can access the API
origins = [
    f"http://127.0.0.1:{os.environ['FRONTEND_PORT']}",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class LeukocyteImagePredictionResponse(BaseModel):
    """
    A response model for leukocyte image prediction
    """
    predictions: dict = {class_name: 0 for class_name in CLASS_MAP}
    prediction_time : float = 0.0


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/model_name", status_code=200)
async def get_model_name():
    """
    Get the name of the loaded model
    """
    return {"model_name": app.state.model_name}


# make this post
@app.get("/predict", response_model=LeukocyteImagePredictionResponse, status_code=200)
async def predict(image_base64_string: Annotated[str | None, Query()] = None, image_path: Annotated[str | None, Query()] = None) -> LeukocyteImagePredictionResponse:
    """
    Make a prediction for a given leukocyte Image
    """

    if image_path is not None and image_base64_string is not None:
        raise ValueError("Must provide either image_base64_string or image_path, not both")
    
    if logging:
        # for debugging the base64 string
        with open('logs/backend_base64.log', 'a') as f:
            f.write(image_base64_string + '\n')

    if image_base64_string is not None:
        raise NotImplementedError("Base64 string prediction is not working yet")
        # make prediction. This is not working for now
        start = time.time()
        image = get_image_tensor_from_base64_string(image_base64_string)

        logits = app.state.model(image.unsqueeze(0)).detach().cpu().squeeze(0)

        predictions = logits.softmax(dim=1)

        return LeukocyteImagePredictionResponse(
            predictions={
                CLASS_MAP_INV[i]: float(predictions[i].item())
                for i in range(len(CLASS_MAP_INV))
            },
            prediction_time=time.time() - start
        )
    
    elif image_path is not None:
        # make predictions
        start = time.time()
        image = get_image_from_path(image_path, device=app.state.device)

        logits = app.state.model(image.unsqueeze(0)).detach().cpu().squeeze(0)

        predictions = logits.softmax(dim=0)

        return LeukocyteImagePredictionResponse(
            predictions={
                CLASS_MAP_INV[i]: float(predictions[i].item())
                for i in range(len(CLASS_MAP_INV))
            },
            prediction_time=time.time() - start
        )

    else:
        return LeukocyteImagePredictionResponse(
            predictions={
                CLASS_MAP_INV[i]: 0.0
                for i in range(len(CLASS_MAP_INV))
            },
            prediction_time=0.0
        )
