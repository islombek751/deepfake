"""
custom_dataset.py

This module provides utility functions for loading image datasets using torchvision's ImageFolder
with a custom transform and filtering logic.

Features:
- Applies resizing and normalization transforms suitable for pre-trained models.
- Filters image files by supported extensions.
- Cleans unwanted Jupyter checkpoint folders ('.ipynb_checkpoints') from dataset directories.
"""

import os
import shutil
from torchvision import datasets, transforms


def get_transform():
    """
    Returns a torchvision transform pipeline for images.

    Transform includes:
    - Resize to 200x200
    - Convert to tensor
    - Normalize with ImageNet mean and std

    Returns:
        torchvision.transforms.Compose: Transformation pipeline.
    """
    return transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def is_valid_file(filename):
    """
    Checks if the file has a valid image extension.

    Args:
        filename (str): Filename to check.

    Returns:
        bool: True if file is an image with supported extension.
    """
    valid_extensions = (
        '.jpg', '.jpeg', '.png', '.ppm', '.bmp',
        '.pgm', '.tif', '.tiff', '.webp'
    )
    return filename.lower().endswith(valid_extensions)


def remove_ipynb_checkpoints(data_dir):
    """
    Removes the '.ipynb_checkpoints' directory inside the dataset folder (if exists).

    Args:
        data_dir (str): Path to the dataset directory.
    """
    checkpoint_path = os.path.join(data_dir, '.ipynb_checkpoints')
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)


def get_dataset(data_dir):
    """
    Loads an image dataset using torchvision's ImageFolder.

    It applies image transforms, filters valid image files, and removes
    unwanted Jupyter checkpoint folders.

    Args:
        data_dir (str): Path to the dataset root directory.

    Returns:
        torchvision.datasets.ImageFolder: The dataset object.
    """
    remove_ipynb_checkpoints(data_dir)
    transform = get_transform()
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform,
        is_valid_file=is_valid_file
    )
    return dataset
