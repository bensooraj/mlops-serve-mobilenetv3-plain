from PIL import Image
import torch
import torchvision.transforms as transforms


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
