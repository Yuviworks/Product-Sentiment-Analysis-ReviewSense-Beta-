import streamlit as st
import joblib
import re
import string
import matplotlib.pyplot as plt
import numpy as np

# =========================================
# 📌 PAGE CONFIG
# =========================================
st.set_page_config(page_title="ReviewSense", layout="centered")

# =========================================
# 🎨 PREMIUM UI + FONT
# =========================================
st.markdown("""
<style>

/* Import Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

/* Apply Font Globally */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Title */
.big-title {
    font-size: 64px;
    font-weight: 700;
    text-align: center;
    color: white;
    letter-spacing: 1px;
    margin-bottom: 5px;
}

/* Subtitle */
.sub-text {
    text-align: center;
    font-size: 20px;
    color: #cbd5e1;
    margin-bottom: 8px;
}

/* Tagline */
.tagline {
    text-align: center;
    font-size: 14px;
    color: #94a3b8;
    margin-bottom: 35px;
}

/* Glass Card */
.glass {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 22px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 20px;
}

/* Buttons */
.stButton>button {
    border-radius: 12px;
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
    font-weight: 600;
    font-size: 16px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.06);
}

/* Textarea */
textarea {
    border-radius: 12px !important;
    font-size: 15px !important;
}

/* Headers */
h2, h3 {
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# 🔥 HEADER
# =========================================
st.markdown('<p class="big-title">🛍️ ReviewSense</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Decode how customers feel about your product</p>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Simple • Fast • Insightful</p>', unsafe_allow_html=True)

# =========================================
# 📌 LOAD MODEL
# =========================================
@st.cache_resource
def load_model():
    model = joblib.load("model/sentiment_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# =========================================
# 📌 CLEAN FUNCTION
# =========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# =========================================
# 📌 SESSION STATE
# =========================================
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# =========================================
# 💡 EXAMPLES
# =========================================
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("💡 Try Examples")

col1, col2, col3 = st.columns(3)

if col1.button("😊 Positive"):
    st.session_state.review_text = "Absolutely amazing product! Works perfectly."

if col2.button("😠 Negative"):
    st.session_state.review_text = "Terrible quality, completely useless."

if col3.button("😐 Neutral"):
    st.session_state.review_text = "It's okay, does the job but nothing special."

st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# ✍️ INPUT
# =========================================
review = st.text_area(
    "✍️ Enter your product review:",
    value=st.session_state.review_text,
    height=150
)

st.session_state.review_text = review

# =========================================
# 🚀 ANALYZE
# =========================================
if st.button("🚀 Analyze Sentiment"):

    if review.strip() == "":
        st.warning("⚠️ Please enter a review!")
    else:
        with st.spinner("Analyzing sentiment..."):
            cleaned = clean_text(review)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        # =========================================
        # 📌 RESULT
        # =========================================
        if prediction == "positive":
            st.markdown("## 😊 Positive")
            st.success("This review expresses a positive sentiment.")

        elif prediction == "negative":
            st.markdown("## 😠 Negative")
            st.error("This review expresses a negative sentiment.")

        else:
            st.markdown("## 😐 Neutral")
            st.info("This review is neutral.")

        # =========================================
        # 📊 CONFIDENCE + CHART
        # =========================================
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vectorized)[0]
            labels = model.classes_

            confidence = max(probs)
            st.progress(float(confidence))
            st.write(f"Confidence: {confidence:.2f}")

            fig, ax = plt.subplots()
            ax.bar(labels, probs)
            ax.set_title("Prediction Probabilities")
            st.pyplot(fig)

        # =========================================
        # 🧠 KEYWORDS
        # =========================================
        st.subheader("🧠 Key Words")

        feature_names = vectorizer.get_feature_names_out()
        vector = vectorized.toarray()[0]

        top_indices = np.argsort(vector)[-5:]
        keywords = [feature_names[i] for i in top_indices if vector[i] > 0]

        st.write(", ".join(keywords) if keywords else "No strong keywords detected")

        st.markdown('</div>', unsafe_allow_html=True)