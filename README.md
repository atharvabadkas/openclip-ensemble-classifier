# 🧠 CLIPClass v2: Multi-Ingredient Classification with Augmented OpenCLIP and Ensemble Learning

CLIPClass v2 is an advanced few-shot image classification system designed for food preparation and waste analysis. Built on **OpenCLIP (ViT-L/14)** and a lightweight **SVM classifier**, it provides high-confidence predictions on small, class-heavy datasets using augmented embeddings and ensemble logic.

---

## 🔍 Overview

CLIPClass v2 improves upon its predecessor with:
- ✅ Class-balanced sampling during training
- 🔁 Robust test-time augmentation (10+ transforms)
- 📊 Ensemble prediction: majority vote + average probability
- 🎯 Top-3 predictions with confidence filtering
- 🗂️ Batch folder inference with real-time progress and CSV output
- 🧩 Exception handling and MPS/CPU support for Apple Silicon

---

### 🔁 Training Pipeline

**Script**: `varandah_prep_vessel_training.py`

1. Loads OpenCLIP (ViT-L/14) with pretrained weights.
2. Applies 10 augmentations per image using Albumentations.
3. Samples a fixed number of images per class for balance.
4. Generates embeddings from augmented images.
5. Trains a linear SVM with class weights.
6. Saves:
   - `prep_ingredient_classifier.pkl`
   - `prep_ingredient_encoder.pkl`


---

## 🔍 Inference Pipeline

**Script**: `varandah_prep_vessel_model.py`

1. Loads trained classifier and label encoder.
2. Applies 10 augmentations to each input image.
3. Generates embeddings via OpenCLIP (ViT-L/14).
4. Makes predictions using:
   - Majority vote
   - Average probability
   - Confidence threshold filtering
5. Outputs predictions and logs results in a CSV.


---

## 🔁 Test-Time Augmentation (TTA)

Augmentations used (via Albumentations):
- Horizontal flip
- Random crop
- Resize + padding
- Color jitter
- Brightness/Contrast change
- Rotation
- Gaussian blur
- CLAHE
- RGB shift
- Grayscale or Sepia conversion

> All augmentations are CPU-compatible and MPS-supported on Apple Silicon (M1/M2).

### 📊 CSV Output Format

Each prediction result is stored in a CSV file with the following columns:

| image_path       | prediction | confidence | top_3                         |
|------------------|------------|------------|-------------------------------|
| image_01.jpg     | tomato     | 0.87       | tomato, onion, garlic         |
| image_02.jpg     | spinach    | 0.91       | spinach, coriander, curryleaf |

**Saved to**: `prediction_results.csv`

### 🧪 Model Performance Summary

#### Training Dataset Accuracy

| Version         | Samples/Class | Accuracy (%) |
|-----------------|----------------|--------------|
| CLIPClass v1    | 5              | 83.2%        |
| CLIPClass v2    | 10             | **90.8%**    |

> Improvements stem from class balancing, test-time augmentation, and ensemble logic.

### 🧪 In Production (Django Stream)

Test results when deployed as a Django-integrated classification service:

| Scenario                     | Accuracy (%) |
|------------------------------|--------------|
| Plate with Chopped Onion     | 92.1         |
| Tray with Mixed Vegetables   | 89.4         |
| Bowl with Curry Base         | 91.6         |
| **Average Accuracy**         | **91.0%**    |

### 🧰 System Requirements

- Python 3.8+
- Apple M1/M2 GPU (MPS) or any CPU
- OpenCLIP (ViT-L/14)
- PyTorch with MPS backend enabled (for Mac)
- Required Python Packages:
  - `open_clip_torch`
  - `scikit-learn`
  - `albumentations`
  - `opencv-python`
  - `pandas`
  - `tqdm`


---

## 💡 Design Highlights

- ⚙️ Modular training and inference code for better maintainability
- 🧪 Ensemble logic improves prediction reliability across augmentations
- 🔍 Confidence-based filtering and top-3 outputs
- 🗂️ Batch folder-level inference for scalability
- 📊 Structured CSV reporting for downstream analysis


