import os
import torch
import torchvision.models as models

# Set environment variable
os.environ["TORCH_MODEL_ZOO"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../pre_trained_model")
)

# Load pretrained model
mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)

# Save model
model_path = os.path.join(
    os.environ["TORCH_MODEL_ZOO"], "mobilenet_v3_large_weights.pt"
)
torch.save(mobilenet_v3_large.state_dict(), model_path)
print(f"Model saved to {model_path}")
