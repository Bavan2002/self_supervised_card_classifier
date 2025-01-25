"""
Self-supervised pretraining script using rotation prediction task.
Trains encoder to learn robust visual features without labeled data.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model_architecture import RotationPredictor
from data_loader import RotationDataset


def train_rotation_model(
 data_path,
 output_path="pretrained_encoder.pth",
 num_epochs=15,
 batch_size=32,
 learning_rate=0.001,
):
 """
 Train encoder using rotation prediction as pretext task.

 Args:
 data_path: Directory containing unlabeled images
 output_path: Where to save pretrained encoder weights
 num_epochs: Number of training epochs
 batch_size: Training batch size
 learning_rate: Optimizer learning rate
 """

 # Setup device
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 print(f"Training on device: {device}")

 try:
 # Load dataset
 dataset = RotationDataset(data_path, target_size=(128, 128), augment=True)
 dataloader = DataLoader(
 dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
 )

 print(f"Loaded {len(dataset)} images for pretraining")

 except (FileNotFoundError, RuntimeError) as error:
 print(f"Dataset error: {error}")
 return

 # Initialize model
 model = RotationPredictor(image_size=(128, 128)).to(device)
 criterion = nn.CrossEntropyLoss()
 optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
 scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

 print("\n" + "=" * 60)
 print("Starting Self-Supervised Pretraining")
 print("=" * 60 + "\n")

 # Training loop
 for epoch in range(num_epochs):
 model.train()
 total_loss = 0.0
 correct_predictions = 0
 total_samples = 0

 for batch_idx, (images, rotation_labels) in enumerate(dataloader):
 images = images.to(device)
 rotation_labels = rotation_labels.to(device)

 # Forward pass
 optimizer.zero_grad()
 predictions = model(images)
 loss = criterion(predictions, rotation_labels)

 # Backward pass
 loss.backward()
 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 optimizer.step()

 # Track metrics
 total_loss += loss.item()
 _, predicted = torch.max(predictions, 1)
 correct_predictions += (predicted == rotation_labels).sum().item()
 total_samples += rotation_labels.size(0)

 # Epoch statistics
 avg_loss = total_loss / len(dataloader)
 accuracy = 100 * correct_predictions / total_samples
 current_lr = optimizer.param_groups[0]["lr"]

 print(
 f"Epoch [{epoch + 1}/{num_epochs}] | "
 f"Loss: {avg_loss:.4f} | "
 f"Accuracy: {accuracy:.2f}% | "
 f"LR: {current_lr:.6f}"
 )

 scheduler.step()

 # Save pretrained encoder
 torch.save(model.encoder.state_dict(), output_path)
 print(f"\n{'=' * 60}")
 print(f"Pretraining complete! Encoder saved to: {output_path}")
 print("=" * 60)


if __name__ == "__main__":
 # Configuration
 UNLABELED_DATA_DIR = "data/unlabeled_documents"
 ENCODER_SAVE_PATH = "models/pretrained_encoder.pth"

 # Create output directory
 os.makedirs("models", exist_ok=True)

 # Run pretraining
 train_rotation_model(
 data_path=UNLABELED_DATA_DIR,
 output_path=ENCODER_SAVE_PATH,
 num_epochs=15,
 batch_size=32,
 learning_rate=0.001,
 )
