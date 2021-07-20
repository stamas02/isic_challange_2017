# -*- coding: utf-8 -*-


"""
The file for the definition of the SelectiveNet and Classifier models.
    Classifier - Class for a EfficientNet Classifier Model.
    SelectiveNet - Class for the SelectiveNet Model using a EfficientNet encoder.
"""

# Built-in/Generic Imports
import os

# Library Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet


class EfficientNetClassifier(nn.Module):
    """
    Class for the Classifier model that uses an EfficientNet encoder with optional drop out.
    """

    def __init__(self, b=0, num_classes=2, drop_rate=0.5, pretrained=True):
        """
        Initiliser for the model that initialises the models layers.
        :param b: The compound coefficient of the EfficientNet model to be loaded.
        :param drop_rate: The drop rate for the optional dropout layers.
        :param pretrained: Boolean if pretrained weights should be loaded.
        """

        # Calls the super for the nn.Module.
        super(EfficientNetClassifier, self).__init__()

        # Sets the drop rate for the dropout layers.
        self.drop_rate = drop_rate

        # Loads the EfficientNet encoder.
        if pretrained:
            self.encoder = EfficientNet.from_pretrained(f"efficientnet-b{str(b)}")
        else:
            self.encoder = EfficientNet.from_name(f"efficientnet-b{str(b)}")

        self.encoder_pool = nn.AdaptiveAvgPool2d(1)

        # Defines a hidden layer.
        self.hidden = nn.Linear(1280, 512)

        # Defines the output layer of the neural network.
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, dropout=False):
        """
        Performs forward propagation with the Classifier.
        :param x: Input image batch.
        :param dropout: Boolean if dropout should be applied.
        :return: A PyTorch Tensor of logits.
        """

        # Performs forward propagation with the encoder.
        x = self.encoder.extract_features(x)
        x = self.encoder_pool(x)
        x = x.view(x.shape[0], -1)

        # Applies dropout to the model is selected.
        if dropout:
            x = F.dropout(x, self.drop_rate)

        # Performs forward propagation with the hidden layer.
        x = self.hidden(x)

        # Applies dropout to the model is selected.
        if dropout:
            x = F.dropout(x, self.drop_rate)

        # Gets the output logits from the output layer.
        return self.classifier(x)

    def save_model(self, path, name, epoch="best"):
        """
        Method for saving the model.
        :param path: Directory path to save the model.
        :param name: The name of the experiment to be saved.
        :param epoch: Integer for the current epoch to be included in the save name.
        """

        # Checks if the save directory exists and if not creates it.
        os.makedirs(path, exist_ok=True)

        # Saves the model to the save directory.
        torch.save(self.state_dict(), os.path.join(path, f"{name}_cnn_{str(epoch)}.pt"))
