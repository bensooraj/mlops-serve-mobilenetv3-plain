import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from .labels import IMAGENET_CATEGORIES


def load_model() -> torch.nn.Module:
    """
    Load MobileNetV3 model with weights from the specified path.
    Returns a torch.nn.Module.
    """
    # we do not specify ``weights``, i.e. create untrained model
    model = models.mobilenet_v3_large()
    model_weights_path = os.path.join(
        os.environ["TORCH_MODEL_ZOO"], "mobilenet_v3_large_weights.pt"
    )
    model.load_state_dict(torch.load(model_weights_path, weights_only=True))
    model.eval()
    return model


def preprocess_image(image: Image.Image):
    """
    Preprocess a Pillow image for MobileNetV3: resize, center crop, convert to tensor, normalize.
    Returns a torch.Tensor.
    """
    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.PILToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    processed_image = preprocess(image)
    if not isinstance(processed_image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(processed_image)}")
    return processed_image.unsqueeze(0)


def classify_image(model: torch.nn.Module, image: torch.Tensor):
    with torch.no_grad():
        output = model(image)

    _, predicted_class = output.max(1)
    print(f"Predicted class index: {predicted_class.item()}")

    predicted_label = IMAGENET_CATEGORIES[predicted_class.item()]
    print(f"Predicted class label: {predicted_label}")
    return {"class_index": predicted_class.item(), "class_name": predicted_label}
