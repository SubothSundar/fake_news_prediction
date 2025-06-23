# app.py

import streamlit as st
import pickle
import numpy as np

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---- Custom Glassmorphism Style ----
st.markdown("""
    <style>
    body {
        background: #000000;
    }
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
    h1, h2 {
        color: #bb86fc;
    }
    .stTextArea > div > textarea {
        background-color: #121212;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Title ----
st.markdown('<div class="glass"><h1> Fake News Detector</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="glass-sub"><h4 style="color:#CCCCFF;">Paste a news article below to check its authenticity.</h4></div>', unsafe_allow_html=True)

# ---- Input Section ----
#st.markdown('<div class="glass">', unsafe_allow_html=True)
news_text = st.text_area("Paste the News Article Text Below:", height=300)

if st.button("Analyze"):
    if news_text.strip() == "":
        st.warning("Please enter some news content.")
    else:
        # Vectorize user input
        vec_text = vectorizer.transform([news_text])

        # Predict class (0 = fake, 1 = real)
        pred_class = model.predict(vec_text)[0]

        # Predict probabilities for both classes
        proba = model.predict_proba(vec_text)[0]       # e.g. [0.15, 0.85]
        confidence = np.max(proba) * 100               # highest prob → %

        # Format result text
        if pred_class == 1:
            st.success(f"✅ This news seems **REAL**.\n\n**Confidence:** {confidence:.2f}%")
        else:
            st.error(f"⚠️ This news seems **FAKE**.\n\n**Confidence:** {confidence:.2f}%")

st.markdown('</div>', unsafe_allow_html=True)
