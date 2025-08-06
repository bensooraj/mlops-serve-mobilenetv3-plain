from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image

app = FastAPI()


class PredictionResponse(BaseModel):
    class_index: int
    class_name: str


@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)) -> PredictionResponse:
    image = Image.open(file.file)

    print(f"Received image size: {image.size}, mode: {image.mode}")
    return PredictionResponse(class_index=42, class_name="goldfish")


# def main():
#     print("Hello from mlops-serve-mobilenetv3-plain!")
# if __name__ == "__main__":
#     main()
