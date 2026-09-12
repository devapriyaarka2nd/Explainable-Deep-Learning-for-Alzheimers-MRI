# Explainable-Deep-Learning-for-Alzheimers-MRI
# Explainable Deep Learning for Alzheimer's Disease Classification from Structural MRI

A 3D deep learning framework for dementia classification from structural
brain MRI using the OASIS-1 dataset, with subject-level evaluation and
Grad-CAM-based visual interpretability.

## Overview

This project investigates whether a lightweight 3D convolutional neural
network can learn discriminative representations from structural MRI
volumes for dementia classification.

The project combines:

- 3D MRI preprocessing
- Patient-level train/validation/test splitting
- 3D CNN classification
- Class-imbalance handling
- Subject-level prediction aggregation
- ROC-AUC and Precision-Recall analysis
- Validation-based threshold selection
- 3D Grad-CAM visualization
- Error analysis using TP, TN, FP and FN cases

## Research Pipeline

MRI Volume
    ↓
Preprocessing
    ↓
Subject-Level Dataset Split
    ↓
3D CNN
    ↓
Dementia Probability
    ↓
Subject-Level Aggregation
    ↓
Classification + Evaluation
    ↓
3D Grad-CAM
    ↓
Interpretability & Error Analysis

## Dataset

This project uses the processed OASIS-1 MRI dataset.

Dataset:
PROCESSED MRI Scans for Alzheimer's Detection — NINAD AITHAL

Original dataset:
OASIS-1 (Open Access Series of Imaging Studies)

The MRI volumes are skull-stripped and spatially normalized to
MNI152 space with 2 mm isotropic resolution.

The dataset contains:

- 436 MRI records
- 416 unique subjects
- 316 non-demented subjects
- 100 dementia subjects

### Label Definition

For the initial binary classification experiment:

- CDR = 0 → Non-Demented
- CDR > 0 → Dementia

CDR is used as the available clinical staging variable in this experiment
and should not be interpreted as a definitive pathological Alzheimer's
disease diagnosis.

## Preprocessing

Each MRI volume is:

1. Loaded using NiBabel
2. Converted to float32
3. Clipped using the 1st and 99th percentile of non-zero brain voxels
4. Z-score normalized within the brain region
5. Retained at the native normalized resolution of:

91 × 109 × 91

No additional spatial resampling was required.

## Model

A lightweight 3D CNN was implemented using PyTorch.

Architecture:

Conv3D(1 → 16)
→ BatchNorm3D
→ ReLU
→ MaxPool3D

Conv3D(16 → 32)
→ BatchNorm3D
→ ReLU
→ MaxPool3D

Conv3D(32 → 64)
→ BatchNorm3D
→ ReLU
→ MaxPool3D

Conv3D(64 → 128)
→ BatchNorm3D
→ ReLU
→ Adaptive Average Pooling

→ Dropout
→ Fully Connected Layer

## Training

Optimizer:
AdamW

Learning rate:
1e-4

Weight decay:
1e-4

Loss:
Binary Cross-Entropy with Logits

Batch size:
2

Class imbalance:
WeightedRandomSampler

## Data Splitting

Splitting was performed at the subject level to prevent multiple MRI
sessions from the same participant from appearing across different
subsets.

Approximate split:

- Training: 70%
- Validation: 15%
- Testing: 15%

No subject overlap was allowed between training, validation and testing.

## Evaluation

The final classification threshold was selected using Youden's J
statistic on the validation set and then fixed before evaluating the
test set.

Subject-level test performance:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.8889 |
| Average Precision | 0.6971 |
| Accuracy | 0.8095 |
| Precision | 0.5882 |
| Sensitivity | 0.6667 |
| Specificity | 0.8542 |
| F1-score | 0.6250 |

Limitations

This is an initial research prototype rather than a clinically validated
diagnostic system.

Important limitations include:

Relatively small number of subjects
Class imbalance
Single-dataset evaluation
No external validation in the current experiment
Binary labeling based on CDR
Coarse spatial resolution of Grad-CAM
Probability calibration requires further investigation





seaborn
scikit-learn
nibabel
torch
torchvision
jupyter
