import os
import torch
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
from contextlib import asynccontextmanager

from .utils import load_model, preprocess_image, classify_image

#
pt_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Set environment variable
    os.environ["TORCH_MODEL_ZOO"] = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../pre_trained_model")
    )
    # Load the ML model
    global pt_model
    pt_model = load_model()
    print("Model loaded!")
    yield
    print("Shutting down")
    # Clean up scripts go below


app = FastAPI(lifespan=lifespan)


class PredictionResponse(BaseModel):
    class_index: int
    class_name: str


@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)) -> PredictionResponse:
    image = Image.open(file.file)
    print(f"Received image size: {image.size}, mode: {image.mode}")
    image_tensor = preprocess_image(image)
    print(f"Image preprocessed: {image_tensor.size()}")

    # Call classify_image and prepare response
    if not isinstance(pt_model, torch.nn.Module):
        raise TypeError(f"Expected torch.Tensor, got {type(pt_model)}")
    result = classify_image(pt_model, image_tensor)

    return PredictionResponse(
        class_index=result["class_index"], class_name=result["class_name"]
    )
