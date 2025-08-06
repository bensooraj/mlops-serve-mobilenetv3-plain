import os
import sys

from PIL import Image
import torch
import torchvision.models as models

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import preprocess_image
from src.labels import IMAGENET_CATEGORIES

# Set environment variable
os.environ["TORCH_MODEL_ZOO"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../pre_trained_model")
)

# Path to the image
image_path = os.path.join(os.path.dirname(__file__), "red_goldfish.webp")

# Load image using Pillow
image = Image.open(image_path)
print(f"Loaded image size: {image.size}, mode: {image.mode}")

# Preprocess image and print tensor dimensions
tensor = preprocess_image(image)
print(f"Preprocessed tensor shape: {tensor.shape}")

#
# we do not specify ``weights``, i.e. create untrained model
model = models.mobilenet_v3_large()

# Save model
model_weights_path = os.path.join(
    os.environ["TORCH_MODEL_ZOO"], "mobilenet_v3_large_weights.pt"
)
model.load_state_dict(torch.load(model_weights_path, weights_only=True))
model.eval()

with torch.no_grad():
    output = model(tensor)

_, predicted_class = output.max(1)
print(f"Predicted class index: {predicted_class.item()}")

predicted_label = IMAGENET_CATEGORIES[predicted_class.item()]
print(f"Predicted class label: {predicted_label}")
