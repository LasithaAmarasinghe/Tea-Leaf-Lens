# 🍃 TeaLeaf Lens: Edge-AI Quality Inspection System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-blueviolet?style=for-the-badge&logo=mlflow)
![EdgeAI](https://img.shields.io/badge/Edge_AI-TFLite-green?style=for-the-badge)

> **Deployment-ready pathology detection system optimized for constrained edge devices, featuring explainable AI (XAI) and full MLOps tracking.**

## 🚀 Project Overview
**TeaLeaf Lens** is a computer vision system designed to automate the quality control process in tea manufacturing. It detects 8 distinct classes of tea leaf pathologies (including *Anthracnose*, *Red Leaf Spot*, and *Algal Leaf*) with high accuracy, capable of running on low-power microcontrollers or mobile devices.

This system focuses on **production constraints**: minimizing model size without sacrificing recall, and ensuring decision transparency through Grad-CAM.

---

## 📊 Key Engineering Results

| Metric | Original (MobileNetV3) | **TeaLeaf Lens (Quantized)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Size** | ~9.2 MB | **1.06 MB** | **9x Compression**  |
| **Format** | FP32 Keras (.h5) | **INT8 TFLite** | Edge Compatible |
| **Accuracy** | 88.4% (Fine-Tuned) | ~87.9% | <1% Drop |

### 🛠️ Tech Stack
* **Core:** TensorFlow/Keras, OpenCV, NumPy
* **Architecture:** MobileNetV3-Small (Transfer Learning + Fine-Tuning)
* **Optimization:** Post-Training Quantization (PTQ)
* **MLOps:** MLflow & DagsHub for experiment tracking
* **Explainability:** Grad-CAM (Gradient-weighted Class Activation Mapping)

---

## 📂 Dataset Information

The model was trained on the **Tea Sickness Dataset** (collected from Kaggle), consisting of small-scale, real-world field imagery.

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Total Images** | 885 | High scarcity challenge |
| **Classes** | 8 | 7 Pathologies + 1 Healthy |
| **Split Strategy** | 80% Train / 20% Val | Stratified split |
| **Preprocessing** | 224x224 px | MobileNetV3 Input Standard |

### 🏷️ Class Labels
The dataset includes the following 8 classes, representing common diseases in tea plantations:
* **Fungal/Bacterial:** *Anthracnose, Algal Leaf, Bird Eye Spot, Brown Blight, Gray Blight, Red Leaf Spot, White Spot*
* **Control:** *Healthy*

### ⚙️ Data Augmentation Strategy
Given the limited dataset size (approx. 110 images per class), aggressive data augmentation was applied during training to prevent overfitting and improve generalization:
* **Geometric:** Random Rotation (±30°), Horizontal Flip, Zoom (20%).
* **Positional:** Width/Height Shifts (20%) to mimic off-center camera framing.

---

## 💡 The "Explainability" Insight (Engineering Spotlight)

During the development phase, the model initially struggled with "Healthy" leaves under direct flash, misclassifying them as diseased.

By implementing **Grad-CAM**, I visualized the model's attention layer and discovered it was triggering on **specular highlights (glare)** caused by camera flash, confusing them with white lesion spots.

![Grad-CAM Debugging](results/result1.png)
*Left: Original Image with Glare. Right: Heatmap showing AI falsely focusing on the reflection.*

**Action Taken:** This insight confirmed that for the production hardware (`TeaRetina`), purely software fixes are insufficient. I recommended a hardware-level intervention: **Polarization filters** on the camera lens to eliminate surface glare, rather than just training on more noisy data.

---

## 🔄 MLOps Pipeline

This project moves beyond "notebook coding" by implementing a full experiment tracking pipeline using **MLflow** hosted on **DagsHub**.

* **Experiment Tracking:** Logs every run's Hyperparameters (Learning Rate, Dropout).
* **Metric Logging:** Automatically tracks `Validation Accuracy` vs. `TFLite File Size`.
* **Artifacts:** Stores the best `.tflite` model version for every run.

---

## 💻 Installation & Usage

### 1. Clone the Repo
```bash
git clone https://github.com/LasithaAmarasinghe/Tea-Leaf-Lens.git
cd TeaLeaf-Lens
pip install -r requirements.txt
```

### 2. Run Training Pipeline
```bash
python train_pipeline.py
```

### 3. MLflow Tracking
```bash
mlflow ui
```
