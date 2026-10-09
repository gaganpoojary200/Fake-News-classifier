# 📰 TruthLens — Fake News Classification System

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**TruthLens** is an AI-powered Fake News Detection web application built using Natural Language Processing (NLP), Scikit-Learn, and Streamlit. It uses TF-IDF feature extraction combined with a Logistic Regression classifier to analyze news article text and classify it as **REAL** or **FAKE**.

---

## ✨ Features

- ⚡ **Blazing Fast NLP Pipeline:** Utilizes Term Frequency-Inverse Document Frequency (TF-IDF) n-gram vectorization with Logistic Regression for fast, reliable classification.
- 🎨 **Modern Glassmorphic UI:** Styled with modern CSS, gradients, typography, and responsive animations on top of Streamlit.
- 🛠️ **Automated Setup Script:** Comes with a built-in `setup_model.py` script that builds, trains, and serializes `model.pkl` and `vectorizer.pkl`.
- 🔌 **Plug & Play Dataset Extensibility:** Easily swap the initial starter dataset with large-scale benchmarks (such as Kaggle's Fake News Dataset).
- 📦 **Lightweight & Portable:** Zero heavy deep learning runtime requirements — runs smoothly on any local or cloud machine.

---

## 🛠️ Tech Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** [Scikit-learn](https://scikit-learn.org/) (TF-IDF Vectorizer, Logistic Regression)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/)
- **Model Serialization:** [Joblib](https://joblib.readthedocs.io/)

---

## 📁 Project Structure

```text
Fake-News-classifier/
├── app.py              # Main Streamlit application with custom glassmorphism styling
├── setup_model.py      # Training and vectorizer generation pipeline
├── model.pkl           # Trained classification model artifact
├── vectorizer.pkl      # Trained TF-IDF vectorizer artifact
├── requirements.txt    # Python package dependencies
├── .gitignore          # Git exclusion rules
└── README.md           # Documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/gaganpoojary200/Fake-News-classifier.git
cd Fake-News-classifier
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train / Initialize the Model
Run the setup script to build and save the model and vectorizer:
```bash
python setup_model.py
```
*(This produces `model.pkl` and `vectorizer.pkl` ready for inference).*

### 5. Launch the Web Application
Start the Streamlit interface:
```bash
streamlit run app.py
```
The app will automatically launch in your browser at `http://localhost:8501`.

---

## 📊 Training with Custom / Large Datasets

TruthLens is ready to be trained on larger real-world datasets (e.g., [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)):

1. Open `setup_model.py`.
2. Replace the initial starter dataset with your dataset file:
   ```python
   # Replace synthetic data with your CSV file:
   df = pd.read_csv("path_to_your_dataset.csv")
   ```
3. Ensure the columns are mapped to text and target label (`REAL` / `FAKE`):
   ```python
   X = df['text']
   y = df['label']
   ```
4. Re-run `python setup_model.py` to retrain and update `model.pkl` and `vectorizer.pkl`.

---

## 👤 Author

Developed by **[Gagan Poojary](https://github.com/gaganpoojary200)**.
