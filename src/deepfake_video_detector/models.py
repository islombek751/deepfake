import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from .xception import xception, xception_concat
import torchvision


def return_pytorch04_xception(pretrained=False):
    """
    Returns the Xception model compatible with PyTorch 0.4+.

    If `pretrained=True`, loads pre-trained weights and adjusts weight shapes 
    where necessary for compatibility.

    Parameters:
        pretrained (bool): Whether to load pre-trained weights.

    Returns:
        nn.Module: Xception model (with or without pre-trained weights).
    """
    model = xception(pretrained=False)
    if pretrained:
        model.fc = model.last_linear
        del model.last_linear
        state_dict = torch.load('/public/liuhonggu/.torch/models/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model.load_state_dict(state_dict)
        model.last_linear = model.fc
        del model.fc
    return model


class TransferModel(nn.Module):
    """
    A transfer learning wrapper for various base models.

    This class wraps a base model (Xception, ResNet18, ResNet50, or custom Xception variant)
    and replaces the final fully connected layer to adapt to binary classification (e.g., real/fake).

    Args:
        modelchoice (str): One of ['xception', 'xception_concat', 'resnet18', 'resnet50'].
        num_out_classes (int): Number of output classes.
        dropout (float): Dropout rate to apply before the final FC layer (default: 0.5).
    """

    def __init__(self, modelchoice, num_out_classes=2, dropout=0.5):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice

        if modelchoice == 'xception':
            self.model = return_pytorch04_xception(pretrained=False)
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )

        elif modelchoice == 'xception_concat':
            self.model = xception_concat()
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )

        elif modelchoice in ['resnet18', 'resnet50']:
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            else:
                self.model = torchvision.models.resnet18(pretrained=True)

            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )

        else:
            raise Exception('Choose a valid model: e.g., resnet50, xception')

    def set_trainable_up_to(self, boolean, layername="Conv2d_4a_3x3"):
        """
        Controls which layers are trainable for fine-tuning.

        Parameters:
            boolean (bool): If True, all layers after `layername` will be trainable.
                            If False, only the final classification layer is trainable.
            layername (str): Name of the layer after which training should begin.
                             If None, all layers are made trainable.
        
        Raises:
            Exception: If the given layer name is not found.
        """
        if layername is None:
            for _, param in self.model.named_parameters():
                param.requires_grad = True
            return
        else:
            for _, param in self.model.named_parameters():
                param.requires_grad = False

        if boolean:
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for param in child.parameters():
                        param.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception(f'Layer "{layername}" not found, can\'t fine-tune!')
        else:
            if self.modelchoice == 'xception':
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True
            else:
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
            x (Tensor): Input image batch.

        Returns:
            Tensor: Raw output (logits) from the final layer.
        """
        return self.model(x)


def model_selection(modelname, num_out_classes, dropout=None):
    """
    Factory method to create the selected model type.

    Parameters:
        modelname (str): One of ['xception', 'xception_concat', 'resnet18'].
        num_out_classes (int): Number of output classes (e.g. 2 for real/fake).
        dropout (float): Dropout rate (optional).

    Returns:
        nn.Module: Initialized TransferModel.

    Raises:
        NotImplementedError: If unknown model name is given.
    """
    if modelname == 'xception':
        return TransferModel(modelchoice='xception', num_out_classes=num_out_classes)

    elif modelname == 'resnet18':
        return TransferModel(modelchoice='resnet18', dropout=dropout, num_out_classes=num_out_classes)

    elif modelname == 'xception_concat':
        return TransferModel(modelchoice='xception_concat', num_out_classes=num_out_classes)

    else:
        raise NotImplementedError(f"Model '{modelname}' is not supported.")
