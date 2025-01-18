"""
Utility functions for model evaluation and visualization.
"""

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np


def evaluate_model(model, dataloader, categories, device="cpu"):
 """
 Evaluate model performance on a dataset.

 Args:
 model: Trained model
 dataloader: DataLoader for evaluation
 categories: List of category names
 device: Device to run evaluation on

 Returns:
 Dictionary with accuracy and per-class metrics
 """
 model.eval()
 model.to(device)

 all_predictions = []
 all_labels = []

 with torch.no_grad():
 for images, labels in dataloader:
 images = images.to(device)
 outputs = model(images)
 _, predicted = torch.max(outputs, 1)

 all_predictions.extend(predicted.cpu().numpy())
 all_labels.extend(labels.numpy())

 # Calculate metrics
 accuracy = 100 * np.mean(np.array(all_predictions) == np.array(all_labels))

 # Generate classification report
 report = classification_report(
 all_labels, all_predictions, target_names=categories, output_dict=True
 )

 return {
 "accuracy": accuracy,
 "predictions": all_predictions,
 "labels": all_labels,
 "report": report,
 }


def plot_confusion_matrix(labels, predictions, categories, save_path=None):
 """
 Plot confusion matrix for model predictions.

 Args:
 labels: True labels
 predictions: Model predictions
 categories: List of category names
 save_path: Optional path to save figure
 """
 cm = confusion_matrix(labels, predictions)

 plt.figure(figsize=(10, 8))
 sns.heatmap(
 cm,
 annot=True,
 fmt="d",
 cmap="Blues",
 xticklabels=categories,
 yticklabels=categories,
 )
 plt.title("Confusion Matrix")
 plt.ylabel("True Label")
 plt.xlabel("Predicted Label")
 plt.tight_layout()

 if save_path:
 plt.savefig(save_path, dpi=300, bbox_inches="tight")
 else:
 plt.show()


def plot_training_history(history, save_path=None):
 """
 Plot training and validation metrics over epochs.

 Args:
 history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists
 save_path: Optional path to save figure
 """
 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

 epochs = range(1, len(history["train_loss"]) + 1)

 # Loss plot
 ax1.plot(epochs, history["train_loss"], "b-", label="Training Loss")
 ax1.plot(epochs, history["val_loss"], "r-", label="Validation Loss")
 ax1.set_title("Model Loss")
 ax1.set_xlabel("Epoch")
 ax1.set_ylabel("Loss")
 ax1.legend()
 ax1.grid(True, alpha=0.3)

 # Accuracy plot
 ax2.plot(epochs, history["train_acc"], "b-", label="Training Accuracy")
 ax2.plot(epochs, history["val_acc"], "r-", label="Validation Accuracy")
 ax2.set_title("Model Accuracy")
 ax2.set_xlabel("Epoch")
 ax2.set_ylabel("Accuracy (%)")
 ax2.legend()
 ax2.grid(True, alpha=0.3)

 plt.tight_layout()

 if save_path:
 plt.savefig(save_path, dpi=300, bbox_inches="tight")
 else:
 plt.show()
