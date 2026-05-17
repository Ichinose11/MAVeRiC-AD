# Clinical-Grade Alzheimer's MRI Classification (PoC)

A robust, enterprise-grade Deep Learning pipeline designed to classify the stages of Alzheimer's Disease from 2D MRI scans. This repository serves as a scaled-down Proof of Concept (PoC) for the MAVeRiC-AD architecture, emphasizing zero-leakage data splitting, model calibration, and explainable AI (XAI).

---

# Problem Statement

Alzheimer's Disease (AD) is characterized by subtle neurodegeneration that begins years before clinical symptoms manifest. Current Deep Learning approaches for AD classification from MRI scans frequently suffer from silent methodological flaws:

1. **Data Leakage**  
   Because consecutive MRI slices from the same patient look nearly identical, random splitting causes the model to memorize patient anatomy rather than learning disease progression.

2. **Overconfidence**  
   Standard Softmax outputs are poorly calibrated, making them untrustworthy in clinical settings.

3. **Black-Box Predictions**  
   Medical professionals cannot rely on AI without understanding where the model is focusing to make its diagnosis.

This project solves these issues by implementing a defensive Machine Learning pipeline that treats data integrity and explainability as first-class citizens.

---

# Pipeline Explanation & Architecture

This pipeline utilizes a ResNet-50 backbone with progressive unfreezing and mixed-precision training. To achieve clinical-grade reliability, the architecture includes the following engineering upgrades:

---

## Perceptual Hashing (pHash) Deduplication

Scans the entire dataset to identify and remove visually similar image variants, preventing augmented duplicates from corrupting the validation set and inflating performance metrics.

---

## Manifold-Aware Clustering (Pseudopatients)

Extracts deep feature embeddings using a frozen ResNet-18 and clusters them using Agglomerative Clustering with Cosine Distance. This groups anatomically similar slices into "pseudopatients."

---

## Stratified Group Splitting

Utilizes `StratifiedGroupKFold` to ensure pseudopatient clusters are never split between training and validation sets, absolutely guaranteeing zero slice-level data leakage while maintaining statistical class balance.

---

## Domain Normalization

Computes true dataset-specific mean and standard deviation matrices rather than relying on generic ImageNet statistics, ensuring optimal convergence for grayscale MRI distributions.

---

## Progressive Unfreezing

Strategically thaws deeper convolutional layers (`layer3`, `layer4`) mid-training while simultaneously reducing the learning rate. This prevents catastrophic forgetting of the backbone's pre-trained edge-detection weights while allowing deeper layers to learn MRI-specific biological features.

---

## Mixed-Precision Training

Utilizes PyTorch's `autocast` and `GradScaler` to execute safe operations in 16-bit floating point precision. This reduces GPU VRAM consumption and accelerates training speeds without sacrificing model accuracy.

---

## Data Pipeline Optimization

Maximizes GPU utilization through:
- Multi-processed background data loading (`num_workers`)
- Page-locked CPU memory (`pin_memory=True`)
- Asynchronous, non-blocking data transfers

---

## Dynamic Learning Rate Scheduling

Implements `ReduceLROnPlateau` to continuously monitor validation metrics and automatically reduce the learning rate when validation loss plateaus, helping the model converge toward a more optimal minimum.

---

## Multi-Class ROC-AUC Evaluation

Abandons misleading accuracy-only metrics in favor of a One-vs-Rest (OvR) Macro ROC-AUC strategy. This ensures the model's discriminative power is evaluated fairly across all four stages of disease progression, even under severe class imbalance conditions.

---

## Temperature Scaling

Uses L-BFGS optimization on the validation set to learn a temperature parameter that softens model logits, producing mathematically calibrated and clinically trustworthy confidence scores.

---

## Explainable AI (Grad-CAM)

Generates gradient-weighted class activation maps that allow clinicians to visually verify whether the model is attending to biologically relevant indicators such as:
- Ventricular enlargement
- Hippocampal atrophy
- Cortical degeneration

rather than irrelevant background noise.

---

## MLOps & Experiment Tracking

Fully integrated with :contentReference[oaicite:0]{index=0} W&B to provide:
- Immutable experiment tracking
- Hyperparameter version control
- Training and validation curve logging
- Hardware utilization monitoring
- Reproducibility and auditability

This satisfies the strict traceability requirements of clinical AI deployments.

---

# Dataset

This project utilizes the Alzheimer's Dataset (4 Class of Images) from :contentReference[oaicite:1]{index=1} Kaggle.

The dataset consists of MRI scans categorized into four distinct stages of Alzheimer's progression:

1. Non-Demented  
2. Very Mild Demented  
3. Mild Demented  
4. Moderate Demented  

> **Note:**  
> The pipeline is specifically designed to ingest the original, unaugmented dataset. All augmentations are strictly controlled within the PyTorch `DataLoader` to prevent train-validation leakage.
