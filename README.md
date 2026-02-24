# 🌿 Plant Disease Detection using Deep Learning

A comprehensive deep learning project for detecting plant diseases using state-of-the-art CNN architectures including MobileNetV2, ResNet50, and EfficientNetB0.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Models](#models)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Documentation](#documentation)

## 🎯 Overview

This project implements multiple deep learning models to classify plant diseases from images. It includes:
- Custom baseline CNN model
- Transfer learning models (MobileNetV2, ResNet50, EfficientNetB0)
- Comparison between preprocessed and raw image approaches
- Interactive Streamlit web application for easy testing
- Comprehensive visualization and evaluation tools

## ✨ Features

- 🤖 **Multiple Model Architectures**: Baseline CNN, MobileNetV2, ResNet50, EfficientNetB0
- 🔄 **Two Training Approaches**: DIP-preprocessed images vs. raw images
- 📊 **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and visualizations
- 🌐 **Web Interface**: Interactive Streamlit app for predictions
- 📈 **Training Monitoring**: Automatic model checkpointing and history tracking
- 🎨 **Rich Visualizations**: Training curves, class distributions, sample predictions

## 🧠 Models

### DIP-Preprocessed Models
- **Baseline CNN**: Custom architecture with multiple conv layers
- **MobileNetV2**: Lightweight model optimized for mobile deployment
- **ResNet50**: Deep residual network for high accuracy
- **EfficientNetB0**: Efficient scaling of network depth, width, and resolution

### Raw Image Models
- Same architectures trained on raw (non-preprocessed) images for comparison

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended
- GPU optional (but recommended for faster training)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection
```

2. **Create virtual environment**
```bash
python -m venv .venv
```

3. **Activate virtual environment**
```bash
# Windows
.\.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📦 Dataset

This project uses the PlantVillage dataset or similar plant disease datasets.

### Download Instructions
1. Download the dataset from [Kaggle PlantVillage Dataset](https://www.kaggle.com/datasets/plant-village)
2. Extract to a folder named `dataset/` in the project root
3. Run preprocessing script:
```bash
python step1_preprocess_dataset.py
```

The preprocessing creates a `preprocessed_data/` directory with train/validation splits.

## 💻 Usage

### Quick Demo (Using Pre-trained Model)

**Option 1: Streamlit Web App (Recommended)**
```bash
streamlit run app.py
```
- Opens in browser at http://localhost:8501
- Upload plant images for instant predictions
- View confidence scores and class probabilities

**Option 2: Command Line Demo**
```bash
python demo_clean.py
```

### Training Models

**Full Pipeline (All Models)**
```bash
python run_all_training.py
```

**Individual Model Training**
```bash
# DIP-preprocessed models
python step2_train_mobilenetv2.py
python step3_train_efficientnetb0.py
python step4_train_resnet50.py

# Raw image models
python raw_train_mobilenetv2.py
python raw_train_efficientnetb0.py
python raw_train_resnet50.py
```

**Multi-model Training**
```bash
python multi_model_training.py
```

### Evaluation

**Evaluate All Models**
```bash
# DIP models
python step5_evaluate_all_models.py

# Raw models
python raw_evaluate_all_models.py
```

**Model Comparison App**
```bash
streamlit run app_comparison.py
```

### Auto Training & Monitoring
```bash
python auto_monitor_and_train.py
```

## 📁 Project Structure

```
DIP Project/
├── 📄 README.md                      # This file
├── 📄 requirements.txt               # Python dependencies
├── 📄 .gitignore                     # Git ignore rules
│
├── 🎯 Main Applications
│   ├── app.py                        # Main Streamlit web app
│   ├── app_comparison.py            # Model comparison app
│   ├── app_dip_models.py            # DIP models demo app
│   ├── app_raw_models.py            # Raw models demo app
│   └── demo_clean.py                # CLI demo script
│
├── 🔧 Training Scripts
│   ├── step1_preprocess_dataset.py  # Data preprocessing
│   ├── step2_train_mobilenetv2.py   # Train MobileNetV2
│   ├── step3_train_efficientnetb0.py# Train EfficientNetB0
│   ├── step4_train_resnet50.py      # Train ResNet50
│   ├── step5_evaluate_all_models.py # Evaluate DIP models
│   ├── multi_model_training.py      # Train multiple models
│   ├── run_all_training.py          # Full training pipeline
│   └── auto_monitor_and_train.py    # Auto training monitor
│
├── 🔬 Raw Models Training
│   ├── raw_train_mobilenetv2.py
│   ├── raw_train_efficientnetb0.py
│   ├── raw_train_resnet50.py
│   ├── raw_evaluate_all_models.py
│   └── run_raw_training_pipeline.py
│
├── 📊 Testing & Evaluation
│   ├── dip_project_training.py      # Baseline CNN training
│   └── dip_project_model_testing.py # Model evaluation
│
├── 📚 Documentation
│   ├── PROJECT_SUMMARY.md           # Complete project summary
│   ├── QUICK_START.md               # Quick start guide
│   └── PRESENTATION_GUIDE.md        # Presentation guidelines
│
├── 📂 Output Directories
│   ├── models/                      # DIP-preprocessed models
│   │   ├── mobilenetv2_best.keras
│   │   ├── resnet50_best.keras
│   │   ├── efficientnetb0_best.keras
│   │   ├── evaluation_results.json
│   │   └── model_comparison.csv
│   │
│   ├── models_raw/                  # Raw image models
│   │   └── (same structure as models/)
│   │
│   ├── output_plots/                # DIP model visualizations
│   └── output_plots_raw/            # Raw model visualizations
│
├── 📂 Data (not in repo - download separately)
│   └── preprocessed_data/
│       ├── train/
│       └── valid/
│
└── 📂 Archive
    └── archive/                     # Old notebooks and files
```

## 📊 Results

### Model Performance (DIP-Preprocessed)
- **MobileNetV2**: Lightweight with good accuracy
- **ResNet50**: High accuracy with deeper architecture
- **EfficientNetB0**: Best balance of accuracy and efficiency

*Detailed results available in `models/evaluation_results.json` after training*

### Comparison: DIP vs Raw
Results comparing preprocessed vs. raw image training approaches are available in:
- `models/model_comparison.csv`
- `models_raw/model_comparison_raw.csv`

## 📚 Documentation

For more detailed information, see:
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Complete project overview and status
- [QUICK_START.md](QUICK_START.md) - Quick start guide for running the project
- [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md) - Guide for presenting the project

## 🛠️ Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Model evaluation metrics

## 📝 License

This project is for educational purposes. Dataset licenses apply separately.

## 🤝 Contributing

This is an academic project. If you have suggestions or improvements:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 👨‍💻 Author

Mukul Yadav - SEM 5 DIP Project

## 🙏 Acknowledgments

- PlantVillage dataset creators
- TensorFlow and Keras teams
- Deep learning research community

---

**Note**: Model files and datasets are not included in this repository due to size constraints. Please download the dataset separately and train models using the provided scripts.
