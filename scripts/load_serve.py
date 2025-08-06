from PIL import Image
import os

import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import preprocess_image

# Path to the image
image_path = os.path.join(os.path.dirname(__file__), "red_goldfish.webp")

# Load image using Pillow

image = Image.open(image_path)
print(f"Loaded image size: {image.size}, mode: {image.mode}")

# Preprocess image and print tensor dimensions
tensor = preprocess_image(image)
print(f"Preprocessed tensor shape: {tensor.shape}")
