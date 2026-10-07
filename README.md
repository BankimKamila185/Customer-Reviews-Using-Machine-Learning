# 🍽️ Restaurant Review Sentiment Analysis Using Machine Learning

[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()
[![Model Accuracy](https://img.shields.io/badge/Test%20Accuracy-83.00%25-brightgreen.svg)]()

> **ITM Skills University | Department of Computer Science and Engineering**  
> **Program:** B.Tech CSE (2024–28) | **Subject:** Machine Learning  
> **Case Study:** Case Study No. 77 – Sentiment Analysis  
> **Author:** Bankim Kamila  

---

## 📌 Project Overview
Online reviews directly impact restaurant reputation and customer decisions. This project delivers an end-to-end **Natural Language Processing (NLP)** and **Supervised Machine Learning** pipeline that automatically classifies raw customer review text into **Positive**, **Negative**, or **Neutral/Mixed** sentiment in real time.

Trained on the benchmark UCI Yelp Restaurant Reviews dataset, the pipeline preprocesses unstructured customer text, extracts statistical word weights via **TF-IDF Vectorization** (1,802 vocabulary features), and evaluates **Logistic Regression** against a **Decision Tree Classifier**. The winning Logistic Regression model (**83.00% Accuracy, 83.33% F1-Score**) is deployed as an interactive web application using **Streamlit**.

---

## 🏗️ System Architecture & Workflow

```
Restaurant Review (dataset.csv)
       │
       ▼
 Data Cleaning (Drop 4 duplicates -> 996 clean rows)
       │
       ▼
      EDA (Sentiment, Rating, Length distributions)
       │
       ▼
Text Preprocessing (Lowercase, regex noise cleaning)
       │
       ▼
    TF-IDF (1,802 Sparse Feature Matrix)
       │
       ▼
Train/Test Split (80% Train: 796 | 20% Test: 200)
       │
 ┌─────┴───────────────┐
 │                     │
Logistic Regression   Decision Tree
 (Acc: 83.00%)         (Acc: 71.50%)
 │                     │
 └─────────┬───────────┘
           │
           ▼
   Model Comparison
           │
           ▼
 Accuracy / Precision / Recall / F1-Score
           │
           ▼
    Confusion Matrix
           │
           ▼
   Best Model Selection (Logistic Regression)
           │
           ▼
 Save Model (model.pkl & vectorizer.pkl)
           │
           ▼
  Streamlit Web Application (app.py)
```

---

## 📊 Key Results & Comparative Performance

Evaluated on $N = 200$ unseen test samples with stratified 50:50 class balance:

| Evaluation Metric | Logistic Regression (Selected Best) | Decision Tree Classifier | Performance Margin ($\Delta$) |
|---|---|---|---|
| **Accuracy** | **83.00%** ($0.8300$) | 71.50% ($0.7150$) | **+11.50%** |
| **Precision** | **81.73%** ($0.8173$) | 70.09% ($0.7009$) | **+11.64%** |
| **Recall** | **85.00%** ($0.8500$) | 75.00% ($0.7500$) | **+10.00%** |
| **F1-Score** | **83.33%** ($0.8333$) | 72.46% ($0.7246$) | **+10.87%** |

### Logistic Regression Confusion Matrix (200 Test Reviews)
- **True Negatives (TN):** 81 reviews (81.0% of actual negatives correctly identified)
- **False Positives (FP):** 19 reviews
- **False Negatives (FN):** 15 reviews
- **True Positives (TP):** 85 reviews (85.0% of actual positives correctly identified)

---

## 🖼️ Visual Evidence & Report Screenshots

The repository includes a dedicated [`visualizations/`](visualizations/) directory with 20 high-resolution (300 DPI) evidence artifacts:

| # | Preview & Image Link | Description | Report Chapter |
|---|---|---|---|
| 01 | [01_raw_dataset.png](visualizations/01_raw_dataset.png) | First 10 authentic records from UCI Yelp dataset before cleaning | Dataset & Data Collection |
| 02 | [02_dataset_information.png](visualizations/02_dataset_information.png) | Metadata card: 1,000 rows, 3 columns, 0 nulls, 4 duplicates | Dataset Overview & Schema |
| 03 | [03_data_cleaning.png](visualizations/03_data_cleaning.png) | Before vs After cleaning table showing duplicate removal & text cleaning | Data Preprocessing |
| 04 | [04_sentiment_distribution.png](visualizations/04_sentiment_distribution.png) | Sentiment class balance bar chart & pie proportion (499 Pos / 497 Neg) | Exploratory Data Analysis |
| 05 | [05_rating_distribution.png](visualizations/05_rating_distribution.png) | Binary satisfaction indicator (`Liked`) and 1-to-5 star rating breakdown | Exploratory Data Analysis |
| 06 | [06_review_length.png](visualizations/06_review_length.png) | Histogram & KDE curves for review word counts & character lengths | Exploratory Data Analysis |
| 07 | [07_text_preprocessing.png](visualizations/07_text_preprocessing.png) | Before vs After text normalization table with 6 genuine customer reviews | NLP Preprocessing Pipeline |
| 08 | [08_tfidf_features.png](visualizations/08_tfidf_features.png) | Sample TF-IDF feature matrix with non-zero token weights | Feature Engineering |
| 09 | [09_train_test_split.png](visualizations/09_train_test_split.png) | 80% Training (796 rows) vs 20% Testing (200 rows) stratified split | Experimental Setup |
| 10 | [10_model_comparison.png](visualizations/10_model_comparison.png) | Grouped bar chart comparing Logistic Regression (83%) vs Decision Tree (71.5%) | Model Comparison |
| 11 | [11_accuracy_summary.png](visualizations/11_accuracy_summary.png) | Model performance summary table card detailing all metrics | Performance Evaluation |
| 12 | [12_confusion_matrix.png](visualizations/12_confusion_matrix.png) | Confusion matrix heatmap for Logistic Regression (TN: 81, TP: 85) | Error Matrix & Diagnostics |
| 13 | [13_classification_report.png](visualizations/13_classification_report.png) | Classification report table (Precision, Recall, F1-Score, Support) | Detailed Metrics |
| 14 | [14_error_analysis.png](visualizations/14_error_analysis.png) | Error analysis table analyzing authentic misclassified test reviews | Error Analysis & Diagnostics |
| 15 | [15_best_model.png](visualizations/15_best_model.png) | Selected best model justification card (Logistic Regression: 83.00% Acc) | Model Selection |
| 16 | [16_streamlit_home.png](visualizations/16_streamlit_home.png) | Streamlit web application home interface screenshot | Web UI Deployment |
| 17 | [17_streamlit_positive.png](visualizations/17_streamlit_positive.png) | Streamlit UI demonstrating real-time **Positive** sentiment prediction | Web UI Deployment |
| 18 | [18_streamlit_negative.png](visualizations/18_streamlit_negative.png) | Streamlit UI demonstrating real-time **Negative** sentiment prediction | Web UI Deployment |
| 19 | [19_streamlit_neutral.png](visualizations/19_streamlit_neutral.png) | Streamlit UI demonstrating **Neutral / Mixed** sentiment prediction | Web UI Deployment |
| 20 | [20_project_workflow.png](visualizations/20_project_workflow.png) | Complete end-to-end architecture diagram from raw data to Streamlit | System Architecture |

---

## 💻 Streamlit Web Application Interface

The interactive web application (`app.py`) provides:
- **Instant Sentiment Classification:** Classifies any custom customer review in real time.
- **Probability Scores:** Displays confidence percentages for positive and negative sentiment.
- **Example Quick Selectors:** One-click sample review buttons for fast testing.
- **Mixed / Neutral Detection:** Identifies balanced reviews with ambiguous sentiment.

![Streamlit Application Positive Inference](visualizations/17_streamlit_positive.png)

---

## 📁 Repository Structure

```
Customer-Reviews-Using-Machine-Learning/
│
├── dataset.csv                  # UCI Yelp benchmark restaurant reviews (1,000 records)
├── swiggy_file.csv              # Benchmark reference restaurant dataset
├── sentiment_analysis.ipynb     # Complete 17-step Jupyter Notebook ML pipeline
├── app.py                       # Streamlit web application for real-time inference
├── generate_visualizations.py   # Python script to regenerate all 20 high-res visual PNGs
├── model.pkl                    # Serialized trained Logistic Regression model
├── vectorizer.pkl               # Serialized TF-IDF vectorizer (vocabulary = 1,802)
├── PROJECT_REPORT.md            # Comprehensive academic project report
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
│
└── visualizations/              # 20 High-resolution (300 DPI) visual evidence PNGs
    ├── 01_raw_dataset.png
    ├── 02_dataset_information.png
    ├── 03_data_cleaning.png
    ├── 04_sentiment_distribution.png
    ├── 05_rating_distribution.png
    ├── 06_review_length.png
    ├── 07_text_preprocessing.png
    ├── 08_tfidf_features.png
    ├── 09_train_test_split.png
    ├── 10_model_comparison.png
    ├── 11_accuracy_summary.png
    ├── 12_confusion_matrix.png
    ├── 13_classification_report.png
    ├── 14_error_analysis.png
    ├── 15_best_model.png
    ├── 16_streamlit_home.png
    ├── 17_streamlit_positive.png
    ├── 18_streamlit_negative.png
    ├── 19_streamlit_neutral.png
    ├── 20_project_workflow.png
    └── README.md                # Visual evidence catalog & report mapping
```

---

## 🚀 Installation & Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/BankimKamila185/Customer-Reviews-Using-Machine-Learning.git
cd Customer-Reviews-Using-Machine-Learning
```

### 2. Create and Activate Virtual Environment (Optional)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook
```bash
jupyter notebook sentiment_analysis.ipynb
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The application will open at `http://localhost:8501`.

### 6. (Optional) Regenerate Visual Evidence Images
```bash
python3 generate_visualizations.py
```

---

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **Data Analysis & Processing:** Pandas, NumPy
- **Natural Language Processing & ML:** Scikit-Learn (TF-IDF, Logistic Regression, Decision Tree, Classification Metrics)
- **Data Visualization:** Matplotlib, Seaborn
- **Model Serialization:** Joblib
- **Web Framework:** Streamlit

---

## 📜 Academic Attribution
- **University:** ITM Skills University
- **Course:** B.Tech Computer Science and Engineering (2024–28)
- **Subject:** Machine Learning (Case Study No. 77)
- **Dataset:** UCI Machine Learning Repository (Yelp Sentiment Dataset)
- **Author:** Bankim Kamila
