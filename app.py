"""
Web application for document classification inference.
Upload images and get real-time predictions.
"""

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
from model_architecture import DocumentClassifier


@st.cache_resource
def load_classification_model(model_path, categories):
    """Load and cache the trained model."""
    num_categories = len(categories)
    model = DocumentClassifier(num_categories=num_categories, freeze_encoder=True)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    )
    model.eval()
    return model


def preprocess_image(image, target_size=(128, 128)):
    """Apply preprocessing transformations to input image."""
    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def predict_document_type(model, image, categories):
    """
    Predict document category from image.

    Args:
        model: Trained classifier model
        image: PIL Image object
        categories: List of category names

    Returns:
        Predicted category name and confidence score
    """
    image_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_category = categories[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    return predicted_category, confidence_score


def main():
    """Main application interface."""
    st.set_page_config(page_title="Document Classifier", layout="centered")

    st.title("Document Type Classifier")
    st.markdown("Upload a document image to identify its type using deep learning.")

    # Configuration
    MODEL_PATH = "models/document_classifier.pth"
    DOCUMENT_CATEGORIES = ["bank_card", "id_card", "visiting_card", "voter_id"]

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a document image",
        type=["png", "jpg", "jpeg", "bmp"],
        help="Supported formats: PNG, JPG, JPEG, BMP",
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Document", use_container_width=True)

        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at: {MODEL_PATH}")
            st.info("Please train the model first using the training scripts.")
            return

        # Make prediction
        with st.spinner("Analyzing document..."):
            try:
                model = load_classification_model(MODEL_PATH, DOCUMENT_CATEGORIES)
                predicted_type, confidence = predict_document_type(
                    model, image, DOCUMENT_CATEGORIES
                )

                # Display results
                st.success("Analysis Complete!")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Document Type", predicted_type.replace("_", " ").title())
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}%")

                # Detailed predictions
                with st.expander("View Detailed Predictions"):
                    image_tensor = preprocess_image(image)
                    with torch.no_grad():
                        outputs = model(image_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

                    for idx, category in enumerate(DOCUMENT_CATEGORIES):
                        prob = probs[idx].item() * 100
                        st.progress(
                            prob / 100,
                            text=f"{category.replace('_', ' ').title()}: {prob:.2f}%",
                        )

            except Exception as error:
                st.error(f"Prediction error: {str(error)}")
    else:
        st.info("Upload an image to begin classification")

    # Display example categories
    st.markdown("### Supported Document Types")
    cols = st.columns(len(DOCUMENT_CATEGORIES))
    for idx, category in enumerate(DOCUMENT_CATEGORIES):
        with cols[idx]:
            st.markdown(f"**{category.replace('_', ' ').title()}**")


if __name__ == "__main__":
    main()
