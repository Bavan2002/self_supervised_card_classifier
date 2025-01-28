# Self-Supervised Document Classifier

A deep learning system for document type classification using self-supervised pretraining and transfer learning.

## Overview

This project implements a two-stage training pipeline:

1. **Self-Supervised Pretraining**: The model learns visual features by predicting rotation angles (0°, 90°, 180°, 270°) of unlabeled document images.
2. **Supervised Fine-Tuning**: The pretrained encoder is fine-tuned on a small labeled dataset to classify document types.

This approach allows the model to learn robust visual representations from unlabeled data, reducing the need for large labeled datasets.

## Features

- Self-supervised learning using rotation prediction
- Transfer learning with frozen encoder
- Data augmentation for improved generalization
- Interactive web interface for inference
- Support for multiple document categories
- Modular and extensible architecture

## Project Structure

```
self_supervised_card_classifier/
├── model_architecture.py # Neural network definitions
├── data_loader.py # Dataset classes
├── train_pretraining.py # Self-supervised pretraining script
├── train_classifier.py # Supervised fine-tuning script
├── app.py # Streamlit web application
├── requirements.txt # Python dependencies
├── README.md # Documentation
├── data/
│ ├── unlabeled_documents/ # Unlabeled images for pretraining
│ └── labeled_documents/ # Labeled images (subdirs = categories)
│ ├── category1/
│ ├── category2/
│ └── ...
└── models/ # Saved model weights
 ├── pretrained_encoder.pth
 └── document_classifier.pth
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU

### Setup

```bash
# Clone or navigate to project directory
cd self_supervised_card_classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Organize your data as follows:

**For pretraining** (unlabeled images):
```
data/unlabeled_documents/
├── image1.jpg
├── image2.jpg
└── ...
```

**For fine-tuning** (labeled images):
```
data/labeled_documents/
├── bank_card/
│ ├── img1.jpg
│ └── img2.jpg
├── id_card/
│ ├── img1.jpg
│ └── img2.jpg
└── ...
```

### 2. Self-Supervised Pretraining

Train the encoder using rotation prediction:

```bash
python train_pretraining.py
```

This saves the pretrained encoder to `models/pretrained_encoder.pth`.

**Configuration options** (edit in script):
- `num_epochs`: Number of training epochs (default: 15)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 0.001)

### 3. Supervised Fine-Tuning

Fine-tune the classifier on labeled data:

```bash
python train_classifier.py
```

This saves the complete model to `models/document_classifier.pth`.

**Configuration options**:
- `num_epochs`: Number of epochs (default: 25)
- `batch_size`: Batch size (default: 8)
- `learning_rate`: Learning rate (default: 0.0005)

### 4. Web Application

Launch the interactive web interface:

```bash
streamlit run app.py
```

Open your browser to the provided URL (typically `http://localhost:8501`) and upload images for classification.

## Model Architecture

### Encoder (ConvolutionalEncoder)
- Input: 128×128 RGB images
- 2 convolutional blocks with batch normalization
- MaxPooling for spatial dimension reduction
- Outputs 32×32×32 feature maps

### Rotation Predictor (Pretraining)
- Encoder + rotation classification head
- 4-way classification (0°, 90°, 180°, 270°)
- Dropout regularization

### Document Classifier (Fine-Tuning)
- Pretrained encoder (frozen)
- Classification head with dropout
- Variable number of output classes

## Training Strategy

1. **Pretraining Phase**:
 - Task: Predict rotation angle
 - Data: Unlabeled images
 - Optimizer: AdamW with weight decay
 - Learning rate scheduling: Cosine annealing

2. **Fine-Tuning Phase**:
 - Task: Document classification
 - Data: Small labeled dataset (80/20 train/val split)
 - Encoder frozen, only train classification head
 - Learning rate scheduling: ReduceLROnPlateau
 - Early stopping based on validation loss

## Customization

### Adding New Document Categories

1. Add category folders to `data/labeled_documents/`
2. Run fine-tuning script (automatically detects categories)
3. Update `DOCUMENT_CATEGORIES` list in `app.py`

### Modifying Architecture

Edit `model_architecture.py`:
- Adjust `base_filters` for model capacity
- Add more convolutional layers
- Change feature dimensions

### Data Augmentation

Modify transformations in `data_loader.py`:
- Adjust augmentation parameters
- Add/remove augmentation techniques

## Performance Tips

- Use GPU for faster training (automatic detection)
- Increase batch size if GPU memory allows
- More unlabeled data improves pretraining
- Data augmentation helps with small labeled datasets
- Unfreeze encoder for final fine-tuning epochs if needed

## Troubleshooting

**ImportError: No module named 'torch'**
- Ensure virtual environment is activated
- Run: `pip install -r requirements.txt`

**CUDA out of memory**
- Reduce batch size
- Use smaller image size
- Train on CPU (slower but functional)

**Low validation accuracy**
- Collect more labeled data
- Increase data augmentation
- Train encoder longer in pretraining phase
- Try unfreezing encoder layers

## Technical Details

- **Framework**: PyTorch
- **Image Processing**: PIL, torchvision
- **Web Interface**: Streamlit
- **Optimization**: AdamW, Adam
- **Normalization**: ImageNet statistics

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

Self-supervised learning techniques inspired by research in representation learning and transfer learning for computer vision.
