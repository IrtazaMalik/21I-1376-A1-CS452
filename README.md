# Deep Learning Assignment No. 01 (CS452)

**Student:** Irtaza Malik
**FAST ID:** i21-1376 
**Course:** CS452 - Deep Learning  

---

## 1. Introduction

This project presents an analysis of **facial expression recognition** and **valence-arousal prediction** using multi-task deep learning.  

We implement two baseline CNN architectures (**ResNet50** and **MobileNetV2**) with shared backbones and separate classification/regression heads, along with a **custom lightweight CNN** for comparison.  

---

## 2. Dataset Analysis

### 2.1 Dataset Overview
- **Total Images:** 3,999 facial images  
- **Annotations:** 15,996 `.npy` files (expression, valence, arousal, landmarks per image)  
- **Expression Classes:** 8 balanced classes (500 each, except class 7 with 499)  
- **Valence-Arousal:** Continuous values in range [-1, 1]  
- **Missing Data:** 0% (no `-2` labels found)  

### 2.2 Data Splits
- **Training:** 2,559 images (64%)  
- **Validation:** 640 images (16%)  
- **Test:** 800 images (20%)  
- Stratified sampling (random seed 42)  

### 2.3 Exploratory Analysis
- Expressions are perfectly balanced.  
- Valence & arousal distributions are approximately normal with slight skew.  

---

## 3. Methodology

### 3.1 Preprocessing Pipeline
- Resize to **224×224**  
- Normalize with **ImageNet mean/std**  
- Augmentations: horizontal flip, ±15° rotation, color jitter, random erasing  
- Landmarks reshaped from 136 → (68,2)  

### 3.2 Model Architectures

#### 3.2.1 ResNet50 Multi-Task
- **Backbone:** ResNet50 (ImageNet pretrained)  
- **Parameters:** 23,528,522  
- **Heads:**  
  - Classification: `Linear(2048 → 8)`  
  - Regression: `Linear(2048 → 2)`  

#### 3.2.2 MobileNetV2 Multi-Task
- **Backbone:** MobileNetV2 (ImageNet pretrained)  
- **Parameters:** 2,236,682  
- **Heads:**  
  - Classification: `Linear(1280 → 8)`  
  - Regression: `Linear(1280 → 2)`  

#### 3.2.3 Custom Light-SE-ResNet
- Depthwise-separable convs + Squeeze-and-Excitation + Residuals  
- **Parameters:** 359,650 (v1), 810,314 (v2)  
- Mobile-friendly with attention mechanisms  

### 3.3 Training Configuration
- **Optimizer:** AdamW (lr=1e-4 ResNet50, 3e-4 MobileNetV2)  
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=2)  
- **Batch Size:** 32  
- **Epochs:** 3–25 (early stopping patience=5)  
- **Loss:** CrossEntropy + λ×MSE (λ=1.0)  
- **Masking:** Invalid valence/arousal masked  

---

## 4. Results

### 4.1 Classification Performance

| Model       | Accuracy | F1-Macro | Cohen's Kappa | ROC-AUC (OvR) | PR-AUC |
|-------------|----------|----------|---------------|---------------|--------|
| ResNet50    | 41.13%   | 39.84%   | 32.50%        | 78.41%        | 37.99% |
| MobileNetV2 | 39.25%   | 38.95%   | 30.62%        | 79.60%        | 40.05% |
| Custom v2   | 26.63%   | 22.47%   | 16.28%        | –             | –      |

### 4.2 Regression Performance

**Valence Prediction**

| Model       | RMSE  | Pearson CORR | SAGR | CCC   |
|-------------|-------|--------------|------|-------|
| ResNet50    | 0.402 | 0.546        | 0.728| 0.480 |
| MobileNetV2 | 0.405 | 0.537        | 0.744| 0.471 |
| Custom v2   | 0.469 | 0.430        | 0.686| 0.401 |

**Arousal Prediction**

| Model       | RMSE  | Pearson CORR | SAGR | CCC   |
|-------------|-------|--------------|------|-------|
| ResNet50    | 0.366 | 0.402        | 0.771| 0.355 |
| MobileNetV2 | 0.385 | 0.335        | 0.746| 0.241 |
| Custom v2   | 0.389 | 0.265        | 0.746| 0.245 |

### 4.3 Krippendorff's Alpha
- **ResNet50:** 0.324  
- **MobileNetV2:** 0.303  

### 4.4 Computational Performance
- **ResNet50:** 81.25 ms inference, 7.72s/step training  
- **MobileNetV2:** ~10× faster inference  

---

## 5. Discussion

### 5.1 Model Comparison
- ResNet50: best overall, accuracy **41.13%**, CCC=0.480  
- MobileNetV2: slightly lower accuracy, but 10× faster  

### 5.2 Continuous Metrics Analysis
- **RMSE:** absolute error → ResNet50 lowest  
- **Pearson CORR:** linear relation → ResNet50 strongest  
- **SAGR:** >70% across all models (good polarity detection)  
- **CCC:** most reliable metric for real-world deployment  

### 5.3 Custom Architecture
- Underperformed vs baselines  
- Likely needs longer training or stronger augmentations  

### 5.4 Error Analysis
- Grad-CAM highlights **eyes & mouth regions**  
- Failures: extreme poses, occlusions, ambiguous classes, lighting issues  

---

## 6. Conclusion

- Multi-task learning improves feature sharing between classification & regression  
- **ResNet50** best-performing model  
- **MobileNetV2** offers good efficiency–accuracy trade-off  
- **CCC** is most reliable for continuous emotion prediction  

**Future Work**
- Stronger augmentations (MixUp, CutMix)  
- Attention mechanisms & Transformers  
- Class imbalance strategies & loss functions  
- Cross-dataset evaluation for generalization  

---

## 7. Visual Results

> *(Add these before submission)*

- **Training Graphs**  
  - ![Training Loss](path/to/loss_curve.png)  
  - ![Training Accuracy](path/to/accuracy_curve.png)  

- **Correct vs Incorrect Predictions**  
  - ![Correct Predictions](path/to/correct_samples.png)  
  - ![Incorrect Predictions](path/to/incorrect_samples.png)  

---

## 8. Submission Files

- `FASTID_Name_A1-CS452.ipynb` – Implementation notebook  
- `best_resnet50.pth`, `best_mobilenetv2.pth` – Checkpoints  
- `outputs/` – Figures & tables  
- `README.md` – Instructions  

---
