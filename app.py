import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/seizure_detector_cnn.keras")

@st.cache_data
def load_data():
    X_test = np.load("data/X_test_small.npy")
    y_test = np.load("data/y_test_small.npy")
    return X_test, y_test

model = load_model()
X_test, y_test = load_data()

st.title("Pediatric EEG Seizure Detection (Demo)")
st.write(
    "This demo uses a 1D CNN trained on the CHB-MIT pediatric EEG dataset "
    "to classify short EEG windows as seizure vs non-seizure. "
    "This is for educational purposes only and not a clinical tool."
)

idx = st.slider(
    "Choose an EEG window index",
    min_value=0,
    max_value=X_test.shape[0] - 1,
    value=0
)

sample = X_test[idx]   # shape (2048, 18)
true_label = int(y_test[idx])

sample_batch = np.expand_dims(sample, axis=0)
pred_prob = float(model.predict(sample_batch, verbose=0)[0][0])
pred_label = int(pred_prob > 0.5)

st.subheader("Prediction")
st.write(f"**True label:** {'Seizure' if true_label == 1 else 'Non-seizure'}")
st.write(f"**Predicted label:** {'Seizure' if pred_label == 1 else 'Non-seizure'}")
st.write(f"**Predicted seizure probability:** {pred_prob:.3f}")

st.subheader("EEG window (Channel 1)")
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(sample[:, 0])
ax.set_xlabel("Time (samples)")
ax.set_ylabel("Amplitude (normalized)")
ax.set_title("Channel 1")
st.pyplot(fig)
