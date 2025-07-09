"""
model.py

This module defines the image classification model based on Microsoft's CvT-13
(Convolutional vision Transformer), extended with a custom classifier head.

Main Components:
- CustomClassifier: A fully connected head with Mish activations and dropout.
- get_model(): Function to instantiate and prepare the full model for training or inference.
"""

import torch
import torch.nn as nn
from transformers import CvtForImageClassification


class CustomClassifier(nn.Module):
    """
    Custom classification head for CvT.

    Architecture:
        Linear(384 -> 256) -> Mish -> BatchNorm -> Dropout(0.5)
        -> Linear(256 -> 128) -> Mish -> BatchNorm -> Dropout(0.3)
        -> Linear(128 -> 2)

    This head is designed to replace the default classifier of the pretrained CvT model.
    """

    def __init__(self):
        super(CustomClassifier, self).__init__()
        self.fc1 = nn.Linear(384, 256)
        self.mish1 = nn.Mish(inplace=False)
        self.norm1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(256, 128)
        self.mish2 = nn.Mish(inplace=False)
        self.norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=0.3)

        self.fc_out = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 384)

        Returns:
            Tensor: Output logits of shape (batch_size, 2)
        """
        x = self.dropout1(self.norm1(self.mish1(self.fc1(x))))
        x = self.dropout2(self.norm2(self.mish2(self.fc2(x))))
        return self.fc_out(x)


def get_model(device: torch.device) -> nn.Module:
    """
    Loads the pretrained CvT-13 model and replaces its classifier with the custom head.

    Args:
        device (torch.device): Target device to load the model on.

    Returns:
        nn.Module: Modified CvT-13 model with CustomClassifier as its head.
    """
    model = CvtForImageClassification.from_pretrained('microsoft/cvt-13')
    model.classifier = CustomClassifier()
    model.to(device)
    return model
