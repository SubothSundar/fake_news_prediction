# Fake News Detector

An intelligent web-based application that detects whether a given news article is real or fake using machine learning and natural language processing (NLP).

Built with Python, scikit-learn, TF-IDF, and a clean Streamlit UI.

---

## Features

* Machine Learning model trained on real-world news datasets
* Text-based news classification using TF-IDF and Logistic Regression
* Confidence score displayed for predictions
* Sleek dark-themed UI with black, blue, and purple glass effects
* Ready for both local execution and deployment

---

## Project Structure


fake_news_detection/
├── app.py                  # Streamlit app
├── model.pkl               # Trained ML model
├── vectorizer.pkl          # Saved TF-IDF vectorizer
├── requirements.txt        # Dependencies
├── README.md               # Project overview
├── data/
│   ├── Fake.csv
│   └── True.csv
└── src/
    └── train_model.py      # Model training script


---

## Installation

### 1. Clone the Repository

```
git clone https://github.com/<your-username>/fake-news-detector.git
cd fake-news-detector
```

### 2. Create a Virtual Environment (optional)

```
python -m venv venv
venv\Scripts\activate     # For Windows
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

## Training the Model

If you want to retrain the model from scratch:

```
python src/train_model.py
```

This will generate:

* model.pkl
* vectorizer.pkl

---

## Running the Streamlit App

```
streamlit run app.py
```

Then open the browser and go to:

```
http://localhost:8501
```

---

## Example Inputs

* "India launches new satellite to monitor air pollution." -> Real News
* "NASA confirms aliens living on the moon" -> Fake News

---

## Model Information

* Model: Logistic Regression
* Feature Extraction: TF-IDF Vectorizer
* Accuracy: Approximately 94 percent on test data
* Libraries used: scikit-learn, nltk, pandas

---


## Acknowledgments

* Kaggle: Fake & Real News Dataset
* scikit-learn
* Streamlit
