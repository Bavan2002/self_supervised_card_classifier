"""
Custom dataset classes for loading and preprocessing document images.
Supports both unlabeled (rotation) and labeled (classification) datasets.
"""

import os
import random
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RotationDataset(Dataset):
 """
 Dataset for self-supervised learning via rotation prediction.
 Randomly rotates images and provides rotation angle as label.
 """

 ROTATION_ANGLES = [0, 90, 180, 270]

 def __init__(self, image_directory, target_size=(128, 128), augment=True):
 if not os.path.isdir(image_directory):
 raise FileNotFoundError(f"Directory not found: {image_directory}")

 self.image_directory = image_directory
 self.target_size = target_size

 # Collect all valid image files
 valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
 self.image_files = [
 os.path.join(image_directory, fname)
 for fname in os.listdir(image_directory)
 if fname.lower().endswith(valid_extensions)
 ]

 if not self.image_files:
 raise RuntimeError(f"No images found in {image_directory}")

 # Base transformations
 transform_list = [
 transforms.Resize(target_size),
 ]

 if augment:
 transform_list.extend(
 [
 transforms.ColorJitter(brightness=0.2, contrast=0.2),
 transforms.RandomHorizontalFlip(p=0.3),
 ]
 )

 transform_list.append(transforms.ToTensor())
 self.base_transform = transforms.Compose(transform_list)

 def __len__(self):
 return len(self.image_files)

 def __getitem__(self, index):
 img_path = self.image_files[index]
 image = Image.open(img_path).convert("RGB")

 # Random rotation for self-supervised task
 rotation_idx = random.randint(0, 3)
 rotation_angle = self.ROTATION_ANGLES[rotation_idx]

 if rotation_angle > 0:
 image = image.rotate(rotation_angle, expand=True)

 image_tensor = self.base_transform(image)
 return image_tensor, rotation_idx


class SupervisedDocumentDataset(Dataset):
 """
 Dataset for supervised classification with labeled document images.
 Expects directory structure: root/category_name/image_files
 """

 def __init__(self, root_directory, target_size=(128, 128), augment=False):
 self.root_directory = root_directory
 self.target_size = target_size

 # Extract category names from subdirectories
 self.categories = sorted(
 [
 dirname
 for dirname in os.listdir(root_directory)
 if os.path.isdir(os.path.join(root_directory, dirname))
 ]
 )

 if not self.categories:
 raise RuntimeError(f"No category subdirectories found in {root_directory}")

 self.category_to_index = {cat: idx for idx, cat in enumerate(self.categories)}

 # Collect all image paths with their labels
 self.samples = []
 for category in self.categories:
 category_path = os.path.join(root_directory, category)
 image_pattern = os.path.join(category_path, "*")

 for img_path in glob.glob(image_pattern):
 if os.path.isfile(img_path):
 self.samples.append((img_path, self.category_to_index[category]))

 if not self.samples:
 raise RuntimeError(f"No labeled images found in {root_directory}")

 # Setup transformations
 transform_list = [transforms.Resize(target_size)]

 if augment:
 transform_list.extend(
 [
 transforms.RandomRotation(15),
 transforms.ColorJitter(brightness=0.3, contrast=0.3),
 transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
 ]
 )

 transform_list.extend(
 [
 transforms.ToTensor(),
 transforms.Normalize(
 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
 ),
 ]
 )

 self.transform = transforms.Compose(transform_list)

 def __len__(self):
 return len(self.samples)

 def __getitem__(self, index):
 img_path, label = self.samples[index]
 image = Image.open(img_path).convert("RGB")
 image_tensor = self.transform(image)
 return image_tensor, label

 def get_categories(self):
 """Return list of category names."""
 return self.categories
