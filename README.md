# Structural Damage Image Classification

**UE23CS352A Machine Learning - Mini Project**

A comprehensive machine learning project implementing multiple classification algorithms to detect structural damage in building images.

## Project Overview

This project builds and compares various machine learning models (both classical and deep learning) to classify structural images as **damaged** or **undamaged**. The work is inspired by the Pacific Earthquake Engineering Research (PEER) Center's initiative for automated structural health monitoring.

## Team Members

- Dhriti Harish Jamadagni - PES1UG23CS185
- [Team Member 2 Name] - [SRN]

**Section:** C Section 
**Problem Statement Number:** 31

## Objective

Develop effective image classification models to:
- Accurately classify structural damage in building images
- Compare performance across multiple ML techniques
- Enable rapid seismic risk assessment of structures

## Dataset

- **Total Images:** 5,913 images (224×224 RGB)
- **Damaged:** 2,727 images (46%)
- **Undamaged:** 3,186 images (54%)
- **Split:** 90% training, 10% validation

### Dataset Characteristics:
- Mix of close-ups (wall sections) and wide-shots (entire buildings)
- Various damage types: cracks, structural failures, collapsed buildings
- Contains non-relevant objects: people, curtains, trees, etc.
- Some images have labeling ambiguity

## 🏗️ Project Structure

```
structural-damage-classification/
│
├── data/
│   ├── damaged/           # Damaged building images
│   └── undamaged/         # Undamaged building images
│
├── main.py                # Main training script
├── utils.py               # Utility functions
├── evaluate.py            # Evaluation and visualization
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── models/               # Saved trained models
│   ├── mobilenet_model.h5
│   └── inceptionv3_model.h5
│
├── results/              # Training results and plots
│   ├── model_results.json
│   └── *.png
│
└── docs/
    ├── writeup.pdf       # One-page project summary
    └── presentation.pptx  # Review presentation
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for deep learning models)
- 8GB+ RAM

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/[your-username]/structural-damage-classification.git
cd structural-damage-classification
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download the dataset:**
   - Download images from PEER Center or provided source
   - Organize into `data/damaged/` and `data/undamaged/` folders

## Dependencies

```
numpy>=1.21.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
matplotlib>=3.4.0
seaborn>=0.11.0
pandas>=1.3.0
```

## Usage

### Training All Models

Run the main training script:
```bash
python main.py
```

This will:
1. Load and preprocess the dataset
2. Train all models (KNN, Logistic Regression, SVM, MobileNet, InceptionV3)
3. Evaluate performance on validation set
4. Generate comparison metrics and visualizations
5. Save trained models

### Training Individual Models

```python
from main import StructuralDamageClassifier

classifier = StructuralDamageClassifier(data_dir='data')
classifier.load_and_preprocess_data()

# Train specific model
classifier.train_inceptionv3(epochs=30)
```

### Evaluation

```bash
python evaluate.py --model inceptionv3 --visualize
```

## Models Implemented

### Classical Machine Learning
1. **K-Nearest Neighbors (k=5)**
   - Simple distance-based classifier
   - Baseline model

2. **Logistic Regression**
   - L2 regularization (C=1.0)
   - Linear decision boundary

3. **Support Vector Machine**
   - RBF kernel
   - Optimal hyperplane separation

### Deep Learning
4. **MobileNetV1**
   - Lightweight CNN architecture
   - Transfer learning from ImageNet
   - Custom fully-connected layers

5. **InceptionV3**
   - Advanced CNN architecture
   - Transfer learning with fine-tuning
   - Multiple layers unfrozen for training

## Results

### Model Performance Summary

| Model | Train Accuracy | Validation Accuracy |
|-------|---------------|---------------------|
| K-Nearest Neighbors | 67.4% | 53.7% |
| Logistic Regression | 99.4% | 54.7% |
| SVM (RBF) | 99.7% | 50.7% |
| MobileNetV1 | 80.0% | 67.0% |
| **InceptionV3** | **81.0%** | **83.0%** |

**Best Model:** InceptionV3 with 83% validation accuracy

### Key Findings

- **CNNs significantly outperform classical ML** on image data
- Classical models show severe overfitting (high train, low val accuracy)
- Transfer learning from ImageNet provides strong baseline
- Data augmentation helps reduce overfitting in deep models
- InceptionV3 achieves best balance of bias and variance

### Confusion Matrix (InceptionV3)
```
                Predicted
              Undamaged  Damaged
Actual
Undamaged       448        86
Damaged         117       349
```

## Methodology

### Data Preprocessing
- Image resizing to 224×224 pixels
- Normalization: pixel values scaled to [-1, 1]
- Train/validation split: 90/10 with stratification

### Data Augmentation
- Horizontal flips
- Width/height shifts (±20%)
- Rotation (±10°)
- Zoom (±15%)

### Transfer Learning Approach
1. Load pre-trained models (ImageNet weights)
2. Freeze early layers (general features)
3. Unfreeze top layers (domain-specific features)
4. Add custom classification head
5. Fine-tune on structural damage dataset

## Challenges Faced

1. **Limited Dataset Size**: Only ~6,000 images for training
2. **Label Ambiguity**: Some images difficult to classify even manually
3. **Class Imbalance**: Slight imbalance (54-46 split)
4. **Overfitting**: Classical models severely overfit
5. **Computational Resources**: Deep models require significant GPU time

## Future Improvements

1. **Data Quality**
   - Correct mislabeled images
   - Remove ambiguous samples
   - Crop irrelevant features

2. **Model Enhancements**
   - Ensemble methods (voting/averaging)
   - Attention mechanisms
   - Custom CNN architecture for structural images

3. **Advanced Techniques**
   - Class activation maps (CAM) for interpretability
   - Multi-class classification (damage severity levels)
   - Object detection for localized damage

4. **Deployment**
   - Web application for real-time inference
   - Mobile app for field assessment
   - API for integration with disaster response systems

## References

1. Pacific Earthquake Engineering Research (PEER) Center. (2018). PEER Hub ImageNet Challenge.
2. Gao, Y., & Mosalam, K. M. (2018). Deep Transfer Learning for Image-Based Structural Damage Recognition.
3. Szegedy, C., et al. (2015). Rethinking the Inception Architecture for Computer Vision.
4. Howard, A., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.
5. CS229 Stanford - Machine Learning Course Materials

## License

This project is created for educational purposes as part of the UE23CS352A Machine Learning course.



---

**Last Updated:** October 8, 2025  
**Submission Date:** October 13, 2025
