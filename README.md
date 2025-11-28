# Pediatric EEG Seizure Detection

Pediatric EEG seizure detection with a 1D CNN and Streamlit demo app.

This project trains a **1D Convolutional Neural Network (CNN)** to detect
epileptic seizures from **pediatric EEG** recordings.

This model is trained on a processed version of the **CHB-MIT Scalp EEG
Database** (pediatric subjects with epilepsy), made available on Kaggle as
[`MIT-CHB_processed`](https://www.kaggle.com/datasets/masahirogotoh/mit-chb-processed).

> ⚠️ **Very Important please:** This project is for research/educational purposes only.
> It is **not** a clinical diagnostic tool.

---

## Project Overview

- **Task:** Classify short EEG windows as  
  - `0` → non-seizure  
  - `1` → seizure
- **Input:** 18-channel EEG windows of length 2048 samples  
  Shape: `(time_steps=2048, channels=18)`
- **Model:** Multichannel **1D CNN**
- **Frameworks:** TensorFlow / Keras, NumPy, Streamlit

A small demo app is provided using **Streamlit** so you can interactively
select an EEG window and see the model’s prediction.

---

## Dataset

Original data:

- **CHB-MIT Scalp EEG Database** – pediatric seizure recordings (PhysioNet)

Processed Kaggle dataset used:

- **Name:** `masahirogotoh/mit-chb-processed`
- **Files used:**
  - `signal_samples.npy` – EEG windows, original shape `(9505, 18, 2048)`
  - `is_sz.npy` – labels `(9505,)`, where `1 = seizure`, `0 = non-seizure`

For the demo app, a **small subset** of the test set is saved as:

- `data/X_test_small.npy`
- `data/y_test_small.npy`

---

## Model

The main model is a **1D Convolutional Neural Network** with the following
architecture (Keras):

- Conv1D(32 filters, kernel size 7, ReLU, padding='same')  
- MaxPooling1D(pool_size=2)  
- Conv1D(64 filters, kernel size 5, ReLU, padding='same')  
- MaxPooling1D(pool_size=2)  
- Conv1D(128 filters, kernel size 3, ReLU, padding='same')  
- GlobalAveragePooling1D  
- Dense(64, ReLU)  
- Dropout(0.3)  
- Dense(1, Sigmoid) → seizure probability

Training details (from Colab notebook):

- Loss: **Binary cross-entropy**  
- Optimizer: **Adam**  
- Batch size: **64**  
- Epochs: **10**  
- Metrics: accuracy, plus precision/recall/F1 on the test set

The trained model is saved as:

- `models/seizure_detector_cnn.keras`
