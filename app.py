import streamlit as st
import joblib
import re

# Page Configuration
st.set_page_config(
    page_title="Restaurant Review Sentiment Analyzer",
    page_icon="🍽️",
    layout="centered"
)

# Text cleaning function (matches training preprocessing pipeline)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load trained model and vectorizer
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'model.pkl' and 'vectorizer.pkl' exist in the project directory.")
        return None, None

model, vectorizer = load_artifacts()

# App Header
st.title("🍽️ Restaurant Review Sentiment Analyzer")
st.caption("ITM Skills University | B.Tech CSE Machine Learning Project | Case Study No. 77")
st.markdown("---")

st.markdown("""
This application uses **Natural Language Processing (NLP)** and a **Logistic Regression** model 
trained on genuine restaurant reviews with **TF-IDF Vectorization** to automatically classify customer sentiment.
""")

# Sample Review Quick Select
st.write("#### ⚡ Quick Try Example Reviews:")
col1, col2, col3 = st.columns(3)
if "review_input" not in st.session_state:
    st.session_state.review_input = ""

if col1.button("🟢 Positive Review", use_container_width=True):
    st.session_state.review_input = "The food was absolutely delicious and the service was warm and attentive!"
if col2.button("🔴 Negative Review", use_container_width=True):
    st.session_state.review_input = "The food was cold, tasteless, and the waiter was extremely rude and slow."
if col3.button("🟡 Neutral Review", use_container_width=True):
    st.session_state.review_input = "The food was average and standard, neither exceptional nor terrible."

# Input section
st.write("### Enter Customer Review:")
user_input = st.text_area(
    label="Review Text",
    value=st.session_state.review_input,
    placeholder="e.g. The food was delicious and the staff were very friendly.",
    height=110,
    label_visibility="collapsed"
)

# Analyze button
if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Please enter a review before analyzing.")
    elif model is None or vectorizer is None:
        st.error("Model is not loaded. Please verify model and vectorizer files.")
    else:
        # Preprocess input text
        cleaned_review = clean_text(user_input)
        
        # Transform using TF-IDF vectorizer
        transformed_input = vectorizer.transform([cleaned_review])
        
        # Predict sentiment & probabilities
        prediction = model.predict(transformed_input)[0]
        probabilities = model.predict_proba(transformed_input)[0]
        classes = list(model.classes_)
        
        pos_idx = classes.index("Positive")
        neg_idx = classes.index("Negative")
        pos_prob = probabilities[pos_idx] * 100
        neg_prob = probabilities[neg_idx] * 100
        
        st.markdown("---")
        st.write("### 📊 Prediction Results:")
        
        # Determine sentiment category
        is_neutral = abs(pos_prob - neg_prob) < 15.0  # Balanced probability indicates neutral/mixed
        
        if is_neutral:
            st.warning("**Predicted Sentiment: NEUTRAL / MIXED 🟡**")
            st.info(f"💡 **Analysis:** The feedback exhibits balanced or moderate sentiment without strong polarization.")
        elif prediction.lower() == "positive":
            st.success(f"**Predicted Sentiment: POSITIVE 🟢**")
            st.info(f"💡 **Analysis:** The customer had a **positive** dining experience! (Confidence: {pos_prob:.2f}%)")
        else:
            st.error(f"**Predicted Sentiment: NEGATIVE 🔴**")
            st.info(f"💡 **Analysis:** The customer had a **negative** dining experience. (Confidence: {neg_prob:.2f}%)")
        
        # Metrics Display
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric(label="Positive Probability", value=f"{pos_prob:.1f}%")
        m_col2.metric(label="Negative Probability", value=f"{neg_prob:.1f}%")
        m_col3.metric(label="Model Used", value="Logistic Regression")
        
        st.caption(f"**Cleaned Text Input:** `{cleaned_review}`")

# Academic Project Footer
st.markdown("---")
st.caption("Machine Learning Classification Pipeline • TF-IDF Vectorizer • Accuracy: 83.00%")
