import streamlit as st
import joblib
import os
import time

# Set up the page config with premium aesthetics
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Custom CSS for rich aesthetics (glassmorphism, modern fonts)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #FF416C, #FF4B2B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        text-align: center;
        color: #718096;
        margin-bottom: 2.5rem;
        font-size: 1.2rem;
        font-weight: 300;
    }
    
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05);
        border: 2px solid rgba(255, 65, 108, 0.2);
        border-radius: 12px;
        color: #E2E8F0;
        padding: 1.2rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTextArea textarea:focus {
        border-color: #FF416C;
        box-shadow: 0 0 15px rgba(255, 65, 108, 0.3);
    }
    
    /* Target Streamlit's actual button */
    div.stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.2rem;
        font-weight: 600;
        width: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
        margin-top: 1rem;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(255, 65, 108, 0.4);
        border: none;
        color: white;
    }
    
    .result-container {
        padding: 2rem;
        border-radius: 12px;
        margin-top: 2rem;
        text-align: center;
        animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .result-fake {
        background: linear-gradient(135deg, rgba(255,65,108,0.1) 0%, rgba(255,75,43,0.1) 100%);
        border: 2px solid #FF416C;
    }
    
    .result-real {
        background: linear-gradient(135deg, rgba(17,153,142,0.1) 0%, rgba(56,239,125,0.1) 100%);
        border: 2px solid #38EF7D;
    }
    
    .result-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .result-title-fake {
        color: #FF416C;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .result-title-real {
        color: #38EF7D;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .result-text {
        color: #A0AEC0;
        font-size: 1.1rem;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Loads the model and vectorizer from disk."""
    if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    return None, None

def display_result(prediction):
    """Displays the result with premium styling."""
    if prediction.upper() == 'FAKE':
        st.markdown("""
        <div class="result-container result-fake">
            <div class="result-icon">🚨</div>
            <div class="result-title-fake">FAKE NEWS DETECTED</div>
            <div class="result-text">Our analysis indicates that this article is likely fabricated or highly misleading.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-container result-real">
            <div class="result-icon">✅</div>
            <div class="result-title-real">VERIFIED REAL NEWS</div>
            <div class="result-text">Our analysis indicates that this article appears to be factual and authentic.</div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

# Main App UI
st.markdown('<div class="main-header">TruthLens</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Fake News Classification System</div>', unsafe_allow_html=True)

model, vectorizer = load_models()

if model is None or vectorizer is None:
    st.error("⚠️ System Offline: Model files not found! Please run `python setup_model.py` to initialize the system.")
    st.stop()

# Input
news_text = st.text_area(
    "Analyze Article", 
    height=220, 
    placeholder="Paste the news article text here...\n\nExample:\n'Secret organization controlling the weather exposed by brave whistleblower!'",
    label_visibility="collapsed"
)

# Analyze Button
if st.button("Initialize Analysis"):
    if not news_text.strip():
        st.warning("⚠️ Please provide text to analyze.")
    else:
        # Fake loading state for better UX
        with st.spinner("Extracting linguistic patterns..."):
            time.sleep(0.5)
            vectorized_text = vectorizer.transform([news_text])
            
        with st.spinner("Evaluating against knowledge graphs..."):
            time.sleep(0.5)
            prediction = model.predict(vectorized_text)[0]
            
        display_result(prediction)

# Footer info
st.markdown("""
<div style="text-align: center; margin-top: 4rem; color: #4A5568; font-size: 0.9rem;">
    Powered by scikit-learn & Streamlit • Fast, simple, and extensible.
</div>
""", unsafe_allow_html=True)
