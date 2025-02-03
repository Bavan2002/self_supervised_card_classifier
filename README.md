# Self-Supervised Document Classifier

A deep learning system for document type classification using self-supervised pretraining and transfer learning.

## Quick Start

```bash
# 1. Setup (using uv - recommended)
uv sync

# 2. Train the model
uv run python train_pretraining.py    # Self-supervised pretraining (~5-10 min)
uv run python train_classifier.py     # Supervised fine-tuning (~10-15 min)

# 3. Launch web app
uv run streamlit run app.py
```

Then open http://localhost:8501 in your browser and upload a document image!

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

- Python 3.10 or higher
- [uv package manager](https://github.com/astral-sh/uv) (recommended for fast installation)
- CUDA-compatible GPU (recommended) or CPU

### Setup

**Option 1: Using uv (Recommended)**
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to project directory
cd self_supervised_card_classifier

# Initialize and sync dependencies
uv sync

# All dependencies will be installed automatically!
```

**Option 2: Using pip**
```bash
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
# Using uv (recommended)
uv run python train_pretraining.py

# Or with pip/venv
python train_pretraining.py
```

This saves the pretrained encoder to `models/pretrained_encoder.pth`.

**Configuration options** (edit in script):
- `num_epochs`: Number of training epochs (default: 15)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 0.001)

**Expected Results:**
- Training completes in ~5-10 minutes on CPU
- Final accuracy should reach ~99-100% on rotation prediction
- Encoder file size: ~26KB

### 3. Supervised Fine-Tuning

Fine-tune the classifier on labeled data:

```bash
# Using uv (recommended)
uv run python train_classifier.py

# Or with pip/venv
python train_classifier.py
```

This saves the complete model to `models/document_classifier.pth`.

**Configuration options**:
- `num_epochs`: Number of epochs (default: 25)
- `batch_size`: Batch size (default: 8)
- `learning_rate`: Learning rate (default: 0.0005)

**Expected Results:**
- Training completes in ~10-15 minutes on CPU
- Best validation accuracy should reach 75-100% (depending on dataset size)
- Model file size: ~33MB

### 4. Web Application

Launch the interactive web interface:

```bash
# Using uv (recommended)
uv run streamlit run app.py

# Or with pip/venv
streamlit run app.py
```

Open your browser to the provided URL (typically `http://localhost:8501`) and upload images for classification.

**Features:**
- Real-time document classification
- Confidence scores for predictions
- Detailed probability breakdown for all categories
- Support for PNG, JPG, JPEG, and BMP formats

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
- Ensure you've run `uv sync` or `pip install -r requirements.txt`
- If using venv, make sure it's activated: `source venv/bin/activate`

**CUDA out of memory**
- Reduce batch size in training scripts
- Use smaller image size (modify `target_size` parameter)
- Train on CPU instead (automatic fallback if CUDA unavailable)

**Low validation accuracy**
- Collect more labeled data (aim for 50+ images per category)
- Increase data augmentation in `data_loader.py`
- Train encoder longer in pretraining phase (increase `num_epochs`)
- Try unfreezing encoder layers for final fine-tuning epochs

**Streamlit app won't start**
- Check if model file exists: `ls models/document_classifier.pth`
- Ensure training completed successfully
- Try: `uv run streamlit run app.py --server.port 8502` (different port)

**uv command not found**
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Or use pip instead (see Installation section)

## Technical Details

- **Framework**: PyTorch 2.0+
- **Image Processing**: PIL, torchvision
- **Web Interface**: Streamlit
- **Optimization**: AdamW (pretraining), Adam (fine-tuning)
- **Normalization**: ImageNet statistics
- **Package Manager**: uv (recommended) or pip

## Project History

This project is a refactored and modularized version of an original Jupyter notebook implementation. The code has been restructured into separate Python modules for better maintainability and production use.

## Performance Notes

- With limited training data (5-20 images per category), expect 75-100% accuracy
- For better results, collect more labeled images (50+ per category recommended)
- The self-supervised pretraining helps significantly when labeled data is scarce
- CPU training is functional but slower than GPU (5-15 minutes vs 1-3 minutes)

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

Self-supervised learning techniques inspired by research in representation learning and transfer learning for computer vision.
