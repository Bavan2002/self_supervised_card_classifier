"""
Supervised fine-tuning script for document classification.
Trains classification head using pretrained encoder.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model_architecture import DocumentClassifier
from data_loader import SupervisedDocumentDataset


def fine_tune_classifier(
 labeled_data_path,
 encoder_weights_path,
 output_model_path="document_classifier.pth",
 num_epochs=25,
 batch_size=8,
 learning_rate=0.0005,
):
 """
 Fine-tune classifier on labeled document dataset.

 Args:
 labeled_data_path: Directory with labeled images (subdirs = categories)
 encoder_weights_path: Path to pretrained encoder weights
 output_model_path: Where to save final model
 num_epochs: Training epochs
 batch_size: Batch size for training
 learning_rate: Learning rate for optimizer
 """

 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 print(f"Training on device: {device}")

 try:
 # Load labeled dataset
 full_dataset = SupervisedDocumentDataset(
 labeled_data_path, target_size=(128, 128), augment=True
 )

 # Split into train/validation
 train_size = int(0.8 * len(full_dataset))
 val_size = len(full_dataset) - train_size
 train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

 train_loader = DataLoader(
 train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
 )
 val_loader = DataLoader(
 val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
 )

 categories = full_dataset.get_categories()
 num_categories = len(categories)

 print(f"Training samples: {train_size}")
 print(f"Validation samples: {val_size}")
 print(f"Categories: {categories}")

 except RuntimeError as error:
 print(f"Dataset loading error: {error}")
 return

 # Initialize model with pretrained encoder
 model = DocumentClassifier(
 num_categories=num_categories,
 pretrained_weights=encoder_weights_path,
 freeze_encoder=True,
 ).to(device)

 criterion = nn.CrossEntropyLoss()
 optimizer = optim.Adam(
 model.classification_head.parameters(), lr=learning_rate, weight_decay=1e-4
 )
 scheduler = optim.lr_scheduler.ReduceLROnPlateau(
 optimizer, mode="min", factor=0.5, patience=3
 )

 print("\n" + "=" * 60)
 print("Starting Supervised Fine-Tuning")
 print("=" * 60 + "\n")

 best_val_loss = float("inf")

 # Training loop
 for epoch in range(num_epochs):
 # Training phase
 model.train()
 train_loss = 0.0
 train_correct = 0
 train_total = 0

 for images, labels in train_loader:
 images, labels = images.to(device), labels.to(device)

 optimizer.zero_grad()
 outputs = model(images)
 loss = criterion(outputs, labels)
 loss.backward()
 optimizer.step()

 train_loss += loss.item()
 _, predicted = torch.max(outputs, 1)
 train_correct += (predicted == labels).sum().item()
 train_total += labels.size(0)

 # Validation phase
 model.eval()
 val_loss = 0.0
 val_correct = 0
 val_total = 0

 with torch.no_grad():
 for images, labels in val_loader:
 images, labels = images.to(device), labels.to(device)
 outputs = model(images)
 loss = criterion(outputs, labels)

 val_loss += loss.item()
 _, predicted = torch.max(outputs, 1)
 val_correct += (predicted == labels).sum().item()
 val_total += labels.size(0)

 # Calculate metrics
 avg_train_loss = train_loss / len(train_loader)
 avg_val_loss = val_loss / len(val_loader)
 train_acc = 100 * train_correct / train_total
 val_acc = 100 * val_correct / val_total

 print(f"Epoch [{epoch + 1}/{num_epochs}]")
 print(f" Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
 print(f" Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

 # Learning rate scheduling
 scheduler.step(avg_val_loss)

 # Save best model
 if avg_val_loss < best_val_loss:
 best_val_loss = avg_val_loss
 torch.save(model.state_dict(), output_model_path)
 print(f" >>> New best model saved!")

 print()

 print("=" * 60)
 print(f"Training complete! Best model saved to: {output_model_path}")
 print(f"Final categories: {categories}")
 print("=" * 60)


if __name__ == "__main__":
 # Configuration
 LABELED_DATA_DIR = "data/labeled_documents"
 ENCODER_WEIGHTS = "models/pretrained_encoder.pth"
 MODEL_SAVE_PATH = "models/document_classifier.pth"

 # Create output directory
 os.makedirs("models", exist_ok=True)

 # Run fine-tuning
 fine_tune_classifier(
 labeled_data_path=LABELED_DATA_DIR,
 encoder_weights_path=ENCODER_WEIGHTS,
 output_model_path=MODEL_SAVE_PATH,
 num_epochs=25,
 batch_size=8,
 learning_rate=0.0005,
 )
