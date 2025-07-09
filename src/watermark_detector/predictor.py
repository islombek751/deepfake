"""
watermark_predictor.py

This module contains:
- ImageDataset: PyTorch Dataset to handle input from image paths, numpy arrays, or PIL images.
- WatermarksPredictor: Class to predict watermark presence in a batch or single image using a trained model.
"""

import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler


def read_image_rgb(path: str) -> Image.Image:
    """
    Opens and converts an image from path to RGB format.

    Args:
        path (str): Path to the image file.

    Returns:
        Image.Image: PIL Image in RGB mode.
    """
    pil_img = Image.open(path)
    pil_img.load()

    if pil_img.format == 'PNG' and pil_img.mode != 'RGBA':
        pil_img = pil_img.convert('RGBA')

    return pil_img.convert('RGB')


class ImageDataset(Dataset):
    """
    Custom dataset for loading images from different sources
    (file paths, numpy arrays, or PIL images) and applying transforms.
    """

    def __init__(self, objects, classifier_transforms):
        """
        Args:
            objects (List[str | np.ndarray | PIL.Image]): List of image inputs.
            classifier_transforms (Callable): Transformations to apply to each image.
        """
        self.objects = objects
        self.classifier_transforms = classifier_transforms

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        obj = self.objects[idx]

        if isinstance(obj, str):
            pil_img = read_image_rgb(obj)
        elif isinstance(obj, np.ndarray):
            pil_img = Image.fromarray(obj)
        elif isinstance(obj, Image.Image):
            pil_img = obj
        else:
            raise TypeError(f"Unsupported type: {type(obj)}")

        return self.classifier_transforms(pil_img).float()


class WatermarksPredictor:
    """
    Runs watermark prediction using a PyTorch classification model.
    """

    def __init__(self, wm_model: torch.nn.Module, classifier_transforms, device: torch.device):
        """
        Args:
            wm_model (torch.nn.Module): Pretrained watermark classification model.
            classifier_transforms (Callable): Preprocessing function for input images.
            device (torch.device): Device to run inference on (CPU or CUDA).
        """
        self.wm_model = wm_model.to(device)
        self.wm_model.eval()
        self.classifier_transforms = classifier_transforms
        self.device = device

    def predict_image(self, pil_image: Image.Image) -> int:
        """
        Predicts a single image.

        Args:
            pil_image (Image.Image): Input image.

        Returns:
            int: Predicted class index (e.g., 0 = no watermark, 1 = watermark).
        """
        input_tensor = self.classifier_transforms(pil_image.convert("RGB")).float().unsqueeze(0)
        with torch.no_grad():
            output = self.wm_model(input_tensor.to(self.device))
            prediction = torch.argmax(output, dim=1).item()
        return prediction

    def run(self, files, num_workers: int = 4, bs: int = 8, pbar: bool = True) -> list:
        """
        Predicts watermark presence in a batch of images.

        Args:
            files (List[str | np.ndarray | PIL.Image]): List of images.
            num_workers (int): Number of workers for DataLoader.
            bs (int): Batch size.
            pbar (bool): Whether to show a progress bar.

        Returns:
            List[int]: List of predicted class indices.
        """
        dataset = ImageDataset(files, self.classifier_transforms)
        loader = DataLoader(
            dataset,
            sampler=SequentialSampler(dataset),
            batch_size=bs,
            drop_last=False,
            num_workers=num_workers
        )
        if pbar:
            loader = tqdm(loader, desc="Predicting")

        results = []
        for batch in loader:
            with torch.no_grad():
                outputs = self.wm_model(batch.to(self.device))
                predictions = torch.argmax(outputs, dim=1).cpu().tolist()
                results.extend(predictions)

        return results
