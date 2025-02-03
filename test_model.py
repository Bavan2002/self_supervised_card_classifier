"""
Quick test script to verify the model works correctly.
"""

import torch
from PIL import Image
from model_architecture import DocumentClassifier
from torchvision import transforms
import glob

# Configuration
MODEL_PATH = "models/document_classifier.pth"
CATEGORIES = ["bank_card", "id_card", "visiting_card", "voter_id"]
TEST_IMAGE_DIRS = "data/labeled_documents"

# Load model
model = DocumentClassifier(num_categories=len(CATEGORIES), freeze_encoder=True)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=True)
)
model.eval()

print("Model loaded successfully!")
print(f"Categories: {CATEGORIES}\n")

# Preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Test on one image from each category
for category in CATEGORIES:
    category_path = f"{TEST_IMAGE_DIRS}/{category}"
    images = glob.glob(f"{category_path}/*.png") + glob.glob(f"{category_path}/*.jpg")

    if images:
        test_image_path = images[0]
        image = Image.open(test_image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        predicted_category = CATEGORIES[predicted_idx.item()]
        confidence_score = confidence.item() * 100

        status = "✓" if predicted_category == category else "✗"
        print(
            f"{status} True: {category:15} | Predicted: {predicted_category:15} | Confidence: {confidence_score:6.2f}%"
        )

print("\nModel testing complete!")
