# Heartbeat Classification

## 1. Task

This project focuses on classifying cardiac arrhythmia using Electrocardiogram (ECG) data. Accurate detection and identification of arrhythmias, such as atrial fibrillation, are essential for reducing the risk of strokes and preventing other severe complications associated with heart rhythm disorders.


## 2. Data

**Disclaimer:** The data, containing sensitive information, is not publicly available.

The dataset consists of ECG signal recordings of different lengths. Signals were preprocessed and features extracted to facilitate classification into four categories (three different types of arrhythmia, and a "normal" class).


## 3. Methodology

### 3.1. Pre-Processing

1. **Feature Extraction**: Utilizes `biosppy` and `neurokit2` libraries for signal processing, extracting morphological and heart rate variability (HRV) features from ECG signals.
2. **Missing Value Handling**: Fills missing (i.e., NaN) feature values with medians.


### 3.2. Model

The XGBoost classifier was employed for arrhythmia classification, with class imbalance addressed by applying sample weights computed to ensure balanced class representation.


## 4. Evaluation

The F-score is used to assess the predictive performance of the model.