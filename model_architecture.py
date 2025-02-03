"""
Neural network architectures for self-supervised card classification.
Implements encoder-decoder pattern with rotation prediction pretraining.
"""

import torch
import torch.nn as nn


class ConvolutionalEncoder(nn.Module):
    """
    Feature extraction module using convolutional layers.
    Designed for 128x128 RGB input images.
    """

    def __init__(self, input_channels=3, base_filters=16):
        super(ConvolutionalEncoder, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                base_filters, base_filters * 2, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.feature_extractor(x)


class RotationPredictor(nn.Module):
    """
    Self-supervised model for rotation angle prediction.
    Used during pretraining phase to learn visual representations.
    """

    def __init__(self, image_size=(128, 128), base_filters=16):
        super(RotationPredictor, self).__init__()

        self.encoder = ConvolutionalEncoder(base_filters=base_filters).feature_extractor

        # Calculate feature map dimensions after convolutions
        feature_dim = (image_size[0] // 4) * (image_size[1] // 4) * (base_filters * 2)

        self.rotation_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 4),  # Predict one of 4 rotation angles
        )

    def forward(self, x):
        features = self.encoder(x)
        predictions = self.rotation_head(features)
        return predictions


class DocumentClassifier(nn.Module):
    """
    Complete classification model with pretrained encoder.
    Fine-tuned for document type recognition.
    """

    def __init__(self, num_categories, pretrained_weights=None, freeze_encoder=True):
        super(DocumentClassifier, self).__init__()

        self.encoder = ConvolutionalEncoder().feature_extractor

        if pretrained_weights:
            self.encoder.load_state_dict(
                torch.load(pretrained_weights, weights_only=True)
            )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(32 * 32 * 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_categories),
        )

    def forward(self, x):
        with (
            torch.no_grad()
            if all(not p.requires_grad for p in self.encoder.parameters())
            else torch.enable_grad()
        ):
            features = self.encoder(x)
        output = self.classification_head(features)
        return output

    def unfreeze_encoder(self):
        """Allow encoder fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
