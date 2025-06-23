# app.py
import streamlit as st
import pickle
import numpy as np
import json
import os
import matplotlib.pyplot as plt

# -------- Load model, vectorizer, and stored accuracy --------
model      = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load accuracy saved during training (see train_model.py)
if os.path.exists("metrics.json"):
    accuracy_val = json.load(open("metrics.json"))["accuracy"] * 100  # %
else:
    accuracy_val = None  # fallback if file not found

# -------- Custom Glassmorphism Style --------
st.markdown(
    """
    <style>
    body { background: #000000; }
    .glass {
        background: rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 2rem;
        color: white;
    }
    h1, h2 { color: #bb86fc; }
    .stTextArea > div > textarea {
        background-color: #121212;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------- Title & Subtitle --------
st.markdown('<div class="glass"><h1>Fake News Detector</h1></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="glass-sub"><h4 style="color:#CCCCFF;">Paste a news article below to check its authenticity.</h4></div>',
    unsafe_allow_html=True,
)

# -------- Input --------
news_text = st.text_area("Paste the News Article Text Below:", height=300)

# -------- Analyse Button --------
if st.button("Analyze"):
    if not news_text.strip():
        st.warning("Please enter some news content.")
    else:
        vec_text  = vectorizer.transform([news_text])
        proba     = model.predict_proba(vec_text)[0]        # [P(fake), P(real)]
        pred_idx  = int(np.argmax(proba))
        confidence = np.max(proba) * 100

        # 1. Show overall test accuracy
        if accuracy_val is not None:
            st.metric("Model test accuracy", f"{accuracy_val:.2f}%")

        # 2. Verdict and confidence
        if pred_idx == 1:
            st.success(f"Result: REAL  ({confidence:.2f}% confidence)")
        else:
            st.error(f"Result: FAKE  ({confidence:.2f}% confidence)")

        # 3. Probability bar-chart
        fig, ax = plt.subplots()
        ax.bar(["Fake", "Real"], proba)
        ax.set_ylim([0, 1])
        ax.set_ylabel("Probability")
        ax.set_title("Class probability distribution")
        for i, v in enumerate(proba):
            ax.text(i, v + 0.02, f"{v*100:.1f}%", ha="center")
        st.pyplot(fig)
