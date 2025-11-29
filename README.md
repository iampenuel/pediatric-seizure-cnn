# Pediatric EEG Seizure Detection

In this project we built and deployed a seizure detection model for **pediatric EEG** data.
We trained a **1D Convolutional Neural Network (CNN)** on a processed CHB-MIT pediatric scalp 
EEG dataset to classify short windows as **seizure** or **non-seizure**, and wrapped the model in a simple Streamlit web app 
---

## Problem Statement

Epileptic seizures in children are usually diagnosed by visually inspecting long EEG
recordings. This process is **time-consuming, subjective, and easy to miss events**,
especially when there are hours of data and only a few seizure segments.

Our goal is to build an **automatic classifier** that can flag seizure segments in
pediatric EEG recordings. Even a simple research prototype can:
- help illustrate how machine learning can support clinicians, and  
- show how deep learning can work with real biomedical signals.

---

## Key Results

- Trained a **1D CNN** on multichannel EEG windows (18 channels, 2048 time steps).  
- Achieved **around 90% accuracy** on a held-out test set of seizure vs non-seizure
  windows (exact metrics and confusion matrix are in the notebook and results files).  
- Visualised training vs validation loss/accuracy to confirm the model is learning
  without severe overfitting in the first 10 epochs.  
- Deployed an **interactive Streamlit app** where users can:
  - select an EEG window,
  - view the model’s predicted seizure probability,
  - and see a plot of the EEG signal for that window.

> ⚠️ This model is a **research/educational demo only**. It is not a clinical diagnostic
> tool and should not be used for real medical decisions.

---

## Methodologies

To accomplish this, we:

1. **Loaded and explored the dataset**  
   - Used NumPy to load pre-segmented EEG windows and seizure labels from `.npy` files.  
   - Checked shapes, class balance, and basic statistics.

2. **Preprocessed the EEG signals**  
   - Transformed data from `(samples, channels, time)` to `(samples, time, channels)`.  
   - Normalized each window to have zero mean and unit variance.  
   - Split data into training and test sets with stratification to keep the
     seizure/non-seizure ratio consistent.

3. **Designed and trained a 1D CNN**  
   - Several Conv1D + MaxPooling layers to learn temporal patterns across 18 channels.  
   - GlobalAveragePooling + Dense layers + Dropout for classification.  
   - Trained for 10 epochs using Adam optimizer and binary cross-entropy loss.

4. **Evaluated the model**  
   - Computed accuracy, precision, recall, F1-score, and a confusion matrix.  
   - Saved training curves and evaluation plots for reporting.

5. **Deployed a demo web app**  
   - Built a Streamlit app (`app.py`) that loads the saved Keras model and a small
     subset of test data.  
   - App runs on Streamlit Cloud and allows interactive inspection of predictions.

---

## Data Sources

- **Primary dataset (processed):**  
  Kaggle – *MIT-CHB_processed*  
  (`masahirogotoh/mit-chb-processed`) – pre-segmented pediatric EEG windows and labels.

- **Original raw dataset:**  
  CHB-MIT Scalp EEG Database (pediatric subjects with intractable seizures).

---

## Technologies Used

- **Languages:** Python  
- **Core libraries:** NumPy, scikit-learn, TensorFlow / Keras, Matplotlib  
- **Tools / Platforms:**  
  - Google Colab (training and experimentation)  
  - Kaggle (dataset hosting)  
  - Streamlit (web app)  
  - Git & GitHub (version control and collaboration)

---

## Authors

This project was completed in collaboration as **Team PYB**:

- *Yuxin Zeng* – *[yu.zeng@tufts.edu]*  
- *Brian Roberto Sugira* – *[rbs23a@acu.edu]*  
- *Penuel Stanley-Zebulon* – *[pcs5301@psu.edu]*  
