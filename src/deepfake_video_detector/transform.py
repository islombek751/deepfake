"""
Author: Andreas Rössler

This module defines standard image preprocessing pipelines (transforms)
for training, validation, and testing deep learning models, specifically
tailored for use with the Xception architecture and similar CNNs.

Each transform pipeline resizes images to a fixed size, converts them to
PyTorch tensors, and normalizes pixel values to the [-1, 1] range using
mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5].
"""

from torchvision import transforms

# Default Xception input size: 299x299
xception_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}
"""
xception_default_data_transforms:
    Standard preprocessing for 299x299 input images.
    Commonly used with Xception models trained on Deepfake datasets.
    Applies to training, validation, and test datasets.
"""

# Alternate size: 256x256 (used for experiments or model variations)
xception_default_data_transforms_256 = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'val': transforms.Resize((256, 256)),  # Note: not converted to tensor
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}
"""
xception_default_data_transforms_256:
    Preprocessing pipeline for 256x256 input size.
    Suitable for alternative models or faster experiments.
    ⚠️ Note: 'val' transform only resizes; does not convert to tensor or normalize.
"""

# Alternate size: 224x224 (commonly used with ResNet, MobileNet, etc.)
transforms_224 = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}
"""
transforms_224:
    Standard preprocessing for models expecting 224x224 input size.
    Typically used with ResNet18/50, MobileNet, VGG, etc.
"""
