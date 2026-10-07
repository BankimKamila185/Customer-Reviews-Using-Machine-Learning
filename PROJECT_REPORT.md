# Machine Learning Academic Project Report
## Restaurant Review Sentiment Analysis Using Machine Learning

---

### Academic Information
- **Institution:** ITM Skills University
- **Department:** Department of Computer Science and Engineering
- **Program:** B.Tech CSE (2024–2028)
- **Course:** Machine Learning
- **Case Study:** Case Study No. 77 – Sentiment Analysis
- **Project Title:** *Restaurant Review Sentiment Analysis Using Machine Learning*
- **Author / Student:** Bankim Kamila
- **Submission Date:** October 2026

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Objectives](#2-problem-statement--objectives)
3. [System Architecture & Workflow](#3-system-architecture--workflow)
4. [Dataset Description & Verification](#4-dataset-description--verification)
5. [Data Cleaning & Text Preprocessing](#5-data-cleaning--text-preprocessing)
6. [Exploratory Data Analysis (EDA)](#6-exploratory-data-analysis-eda)
7. [Feature Engineering (TF-IDF Vectorization)](#7-feature-engineering-tf-idf-vectorization)
8. [Experimental Setup & Dataset Partitioning](#8-experimental-setup--dataset-partitioning)
9. [Machine Learning Models & Training](#9-machine-learning-models--training)
10. [Comparative Model Evaluation & Results](#10-comparative-model-evaluation--results)
11. [Confusion Matrix & Classification Metrics](#11-confusion-matrix--classification-metrics)
12. [Error Analysis & Qualitative Diagnostics](#12-error-analysis--qualitative-diagnostics)
13. [Selected Best Model Justification](#13-selected-best-model-justification)
14. [Interactive Streamlit Web Application](#14-interactive-streamlit-web-application)
15. [Limitations & Future Scope](#15-limitations--future-scope)
16. [Conclusion](#16-conclusion)

---

## 1. Executive Summary

With the explosive growth of online dining review platforms such as Yelp, Google Reviews, and Zomato, automated text sentiment classification has become essential for restaurants to monitor customer satisfaction and make data-driven operational decisions. 

This project implements an end-to-end Supervised Machine Learning and Natural Language Processing (NLP) system to classify restaurant reviews into sentiment classes. Starting with the UCI Yelp benchmark dataset of 1,000 real customer reviews, data cleaning, regex-based text normalization, exploratory data analysis, and TF-IDF feature extraction (1,802 unique tokens) were performed. Two supervised classification models—**Logistic Regression** and **Decision Tree Classifier**—were trained and evaluated on 200 unseen test records. 

**Key Findings:**
- **Logistic Regression** achieved an overall test accuracy of **83.00%**, precision of **81.73%**, recall of **85.00%**, and an F1-score of **83.33%**.
- **Decision Tree Classifier** achieved an accuracy of **71.50%** and an F1-score of **72.46%**, struggling with overfitting on high-dimensional sparse text data.
- Logistic Regression was selected as the optimal production model and deployed as a real-time web application using **Streamlit**.

---

## 2. Problem Statement & Objectives

### Problem Statement
Customer feedback text contains unstructured natural language featuring varied syntax, slang, mixed emotions, and negation. Manual analysis of thousands of customer reviews is prohibitively expensive and slow. The objective is to build an automated, interpretable, and accurate machine learning pipeline capable of parsing raw customer reviews and predicting dining sentiment in real time.

### Objectives
1. Formulate the sentiment analysis problem as a supervised text classification task.
2. Ingest and document a genuine benchmark restaurant reviews dataset.
3. Apply standard text preprocessing techniques (lowercasing, punctuation removal, regex cleaning).
4. Perform exploratory data analysis to inspect class balance, rating indicators, and review length distributions.
5. Extract statistical word weights using **Term Frequency - Inverse Document Frequency (TF-IDF)** without data leakage.
6. Train and rigorously compare **Logistic Regression** versus **Decision Tree Classifier**.
7. Evaluate performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
8. Conduct qualitative error analysis on misclassified samples.
9. Deploy the best-performing model into an interactive **Streamlit web application**.

---

## 3. System Architecture & Workflow

The end-to-end architecture follows a modular, reproducible pipeline:

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

### Visual Evidence: System Architecture Diagram
![End-to-End Project Workflow](visualizations/20_project_workflow.png)

---

## 4. Dataset Description & Verification

The project utilizes the benchmark **UCI Machine Learning Repository Yelp Restaurant Reviews Dataset**. 

### Raw Dataset Preview
The raw dataset contains 1,000 records across three primary columns:
- `Review`: Raw customer text feedback.
- `Liked`: Binary satisfaction rating indicator ($1 = \text{Liked}, 0 = \text{Disliked}$).
- `Sentiment`: Ground-truth categorical target label (`Positive` / `Negative`).

### Visual Evidence: Raw Dataset (First 10 Records)
![Raw Dataset Before Cleaning](visualizations/01_raw_dataset.png)

### Dataset Technical Summary
- **Total Records:** 1,000 records
- **Total Columns:** 3 attributes
- **Missing / Null Values:** 0 (0.0% missing data)
- **Duplicate Records:** 4 records (0.4%)
- **Initial Class Distribution:** 500 Positive, 500 Negative (Perfect 50:50 balance)

### Visual Evidence: Dataset Information & Metadata Card
![Dataset Information](visualizations/02_dataset_information.png)

---

## 5. Data Cleaning & Text Preprocessing

### Data Cleaning
1. **Deduplication:** 4 duplicate review records were removed using `df.drop_duplicates(subset=['Review'])`, yielding 996 unique reviews.
2. **Missing Value Verification:** Zero null or NaN values were confirmed across all columns.
3. **Cleaned Class Distribution:** 499 Positive reviews (50.1%) and 497 Negative reviews (49.9%), preserving class balance.

### Visual Evidence: Data Cleaning Summary & Transformation Table
![Data Cleaning Before vs After](visualizations/03_data_cleaning.png)

### Text Preprocessing Pipeline
Raw natural language text was normalized through the following sequential NLP steps:
1. **Lowercasing:** Standardizes all tokens to lower case (`"FOOD"` $\rightarrow$ `"food"`).
2. **URL Stripping:** Removes any web links or URLs using regular expressions.
3. **Punctuation & Special Character Removal:** Strips symbols, quotation marks, ellipsis, numbers, and emojis using `re.sub(r'[^a-zA-Z\s]', '', text)`.
4. **Whitespace Normalization:** Collapses multiple spaces and trims leading/trailing whitespace.

### Preprocessing Code Snippet
```python
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

### Visual Evidence: Text Preprocessing Comparison
![Text Preprocessing Table](visualizations/07_text_preprocessing.png)

---

## 6. Exploratory Data Analysis (EDA)

### 1. Sentiment Class Balance
The cleaned dataset contains 499 Positive (50.1%) and 497 Negative (49.9%) reviews. This balanced distribution eliminates the need for synthetic oversampling (SMOTE) or class weighting.

### Visual Evidence: Sentiment Distribution
![Sentiment Distribution](visualizations/04_sentiment_distribution.png)

### 2. Rating Distribution Analysis
The dataset contains binary satisfaction indicators (`Liked = 0` vs `Liked = 1`) corresponding to negative and positive restaurant experiences. Mapped to standard 1–5 star reviews, 49.9% represent 1–2 star experiences while 50.1% represent 4–5 star dining experiences.

### Visual Evidence: Rating Distribution
![Rating Distribution](visualizations/05_rating_distribution.png)

### 3. Review Length Distribution Analysis
Analyzing word counts and character lengths revealed:
- **Mean Review Length:** 10.9 words (58.3 characters)
- **Median Review Length:** 10.0 words
- **Word Count Range:** 1 word to 32 words
- Both positive and negative reviews display similar length profiles, confirming length does not introduce classification bias.

### Visual Evidence: Review Length Distribution
![Review Length Distribution](visualizations/06_review_length.png)

---

## 7. Feature Engineering (TF-IDF Vectorization)

### Mathematical Formulation
**TF-IDF (Term Frequency - Inverse Document Frequency)** assigns statistical importance weights to words:

$$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$

$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

### Vectorization Properties
- **Vocabulary Size:** 1,802 unique n-gram features
- **Matrix Sparsity:** > 99% sparse representation
- **No Data Leakage:** The vectorizer was fitted strictly on the training set and applied as a transform on the test set.

### Visual Evidence: Sample TF-IDF Feature Matrix
![Sample TF-IDF Feature Matrix](visualizations/08_tfidf_features.png)

---

## 8. Experimental Setup & Dataset Partitioning

The cleaned dataset ($N = 996$) was partitioned into training and testing subsets using stratified random sampling (`random_state=42`):
- **Training Set (80%):** 796 reviews (399 Positive, 397 Negative)
- **Testing Set (20%):** 200 reviews (100 Positive, 100 Negative)

### Visual Evidence: Train / Test Partitioning Architecture
![Train Test Split](visualizations/09_train_test_split.png)

---

## 9. Machine Learning Models & Training

Two supervised classification algorithms representing distinct learning paradigms were implemented:

### Model 1: Logistic Regression (Linear Probabilistic Classifier)
- **Principle:** Fits a linear decision boundary using the sigmoid function:
  $$P(Y=1|X) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- **Hyperparameters:** `random_state=42`, L2 regularization ($C=1.0$), L-BFGS solver.
- **Strength in NLP:** High stability, convex optimization, well-suited for high-dimensional sparse TF-IDF text matrices.

### Model 2: Decision Tree Classifier (Non-Linear Tree-Based Classifier)
- **Principle:** Recursively partitions the feature space based on Gini Impurity reduction:
  $$\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2$$
- **Hyperparameters:** `random_state=42`, standard CART splitting.
- **Strength in NLP:** Non-parametric, interpretable conditional rules; however, prone to overfitting in sparse vocabulary spaces.

---

## 10. Comparative Model Evaluation & Results

Both models were evaluated on the 200 unseen test records.

### Model Performance Comparison Table

| Machine Learning Model | Accuracy | Precision | Recall | F1-Score | Decision / Status |
|---|---|---|---|---|---|
| **Logistic Regression** | **83.00%** | **81.73%** | **85.00%** | **83.33%** | **Selected Best Model** |
| **Decision Tree Classifier** | **71.50%** | **70.09%** | **75.00%** | **72.46%** | Baseline Model |
| **Performance Difference ($\Delta$)** | **+11.50%** | **+11.64%** | **+10.00%** | **+10.87%** | Logistic Regression Wins Across All Metrics |

### Visual Evidence: Model Performance Comparison Bar Chart
![Model Comparison](visualizations/10_model_comparison.png)

### Visual Evidence: Model Accuracy Summary Card
![Accuracy Summary](visualizations/11_accuracy_summary.png)

---

## 11. Confusion Matrix & Classification Metrics

### Confusion Matrix Breakdown (Logistic Regression)
Evaluated on the 200 test samples (100 Negative, 100 Positive):
- **True Negatives (TN):** **81** reviews correctly classified as Negative (81.0%)
- **False Positives (FP):** **19** reviews incorrectly classified as Positive (19.0%)
- **False Negatives (FN):** **15** reviews incorrectly classified as Negative (15.0%)
- **True Positives (TP):** **85** reviews correctly classified as Positive (85.0%)

### Visual Evidence: Confusion Matrix Heatmap
![Confusion Matrix](visualizations/12_confusion_matrix.png)

### Classification Report (Logistic Regression)

| Sentiment Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Negative** | 0.8438 (84.38%) | 0.8100 (81.00%) | 0.8265 (82.65%) | 100 reviews |
| **Positive** | 0.8173 (81.73%) | 0.8500 (85.00%) | 0.8333 (83.33%) | 100 reviews |
| **Accuracy** | — | — | **0.8300 (83.00%)** | 200 reviews |
| **Macro Average** | 0.8305 (83.05%) | 0.8300 (83.00%) | 0.8299 (82.99%) | 200 reviews |
| **Weighted Average** | 0.8305 (83.05%) | 0.8300 (83.00%) | 0.8299 (82.99%) | 200 reviews |

### Visual Evidence: Classification Report Table
![Classification Report](visualizations/13_classification_report.png)

---

## 12. Error Analysis & Qualitative Diagnostics

Out of 200 test samples, 34 reviews were misclassified. A diagnostic review of authentic test errors revealed three main root causes:

1. **Idiomatic Negation & Colloquialisms:**
   - *"You can't beat that."* (Actual: Positive $\rightarrow$ Predicted: Negative) — The presence of the unigram token *"cant"* triggered a negative score despite the idiom conveying strong satisfaction.
2. **Double Negation & Contrastive Syntax:**
   - *"I found this place by accident and I could not be happier."* (Actual: Positive $\rightarrow$ Predicted: Negative) — The unigram model penalized *"not"* without recognizing the positive meaning of *"not happier"*.
3. **Factual Sentences Lacking Explicit Polarity Adjectives:**
   - *"I was seated immediately."* (Actual: Positive $\rightarrow$ Predicted: Negative) — Highly factual positive service description lacking emotional keywords like *"great"* or *"delicious"*.

### Visual Evidence: Error Analysis Table
![Error Analysis](visualizations/14_error_analysis.png)

---

## 13. Selected Best Model Justification

**Logistic Regression** was selected as the optimal production model based on the following justifications:
1. **Superior Generalization:** Outperformed Decision Tree by +11.50% in test accuracy (83.00% vs 71.50%).
2. **High Precision (81.73%):** Minimizes false-positive alerts, ensuring restaurant managers receive reliable positive/negative signals.
3. **Calibrated Probabilities:** Outputs continuous confidence scores ($0.0\%$ to $100.0\%$) via the sigmoid function, facilitating threshold tuning and neutral/mixed sentiment detection.
4. **Computational Efficiency:** Sub-millisecond inference time enables seamless real-time execution in web applications.

### Visual Evidence: Best Model Decision Card
![Best Model Selection](visualizations/15_best_model.png)

---

## 14. Interactive Streamlit Web Application

The trained model (`model.pkl`) and TF-IDF vectorizer (`vectorizer.pkl`) were integrated into an interactive web application built with **Streamlit** (`app.py`).

### Web App Features
- **Real-Time NLP Inference:** User inputs custom review text; the app preprocesses text, extracts TF-IDF weights, and predicts sentiment.
- **Probability Breakdown:** Displays positive and negative class probabilities alongside model metadata.
- **Quick Preset Selectors:** One-click buttons to test representative positive, negative, and neutral reviews.
- **Neutral / Mixed Sentiment Detection:** Detects balanced feedback when confidence margin is narrow ($|P_{\text{pos}} - P_{\text{neg}}| < 15\%$).

### Visual Evidence: Streamlit Application Screenshots

#### 1. Home / Default State
![Streamlit Home Page](visualizations/16_streamlit_home.png)

#### 2. Positive Review Sentiment Prediction
![Streamlit Positive Prediction](visualizations/17_streamlit_positive.png)

#### 3. Negative Review Sentiment Prediction
![Streamlit Negative Prediction](visualizations/18_streamlit_negative.png)

#### 4. Neutral / Mixed Review Prediction
![Streamlit Neutral Prediction](visualizations/19_streamlit_neutral.png)

---

## 15. Limitations & Future Scope

### Limitations
1. **Unigram TF-IDF Limitations:** Single-word tokenization does not capture long-range contextual dependencies or negation scope (e.g., *"not bad"* vs *"not good"*).
2. **Sarcasm Detection:** Sarcastic feedback (e.g., *"Another cold pizza, brilliant."*) cannot be detected without deep contextual representations.
3. **Domain Vocabulary Specificity:** Regional culinary terms or misspellings absent from the training set are treated as out-of-vocabulary (OOV).

### Future Scope
1. **N-Gram Integration:** Incorporate bi-grams and tri-grams (`ngram_range=(1, 3)`) to resolve negation phrases.
2. **Transformer Fine-Tuning:** Fine-tune pre-trained models such as BERT or RoBERTa for deep contextual understanding.
3. **Aspect-Based Sentiment Analysis (ABSA):** Decompose reviews into separate sub-ratings for food quality, staff service, ambiance, and price value.
4. **Live API Ingestion:** Connect to Google Maps and Zomato APIs for live review ingestion and sentiment tracking dashboards.

---

## 16. Conclusion

This project successfully fulfilled all academic requirements for ITM Skills University Case Study No. 77. An end-to-end supervised machine learning pipeline was constructed, evaluated, and deployed.

### Summary of Achievements
- Curated and validated a benchmark dataset of 1,000 Yelp customer reviews (996 clean records, 0 nulls).
- Implemented robust regex-based NLP text preprocessing and extracted 1,802 TF-IDF features.
- Compared **Logistic Regression** (83.00% Accuracy, 83.33% F1-score) against **Decision Tree** (71.50% Accuracy).
- Generated and cataloged 20 high-resolution visual evidence artifacts.
- Successfully deployed the selected Logistic Regression model as a functional **Streamlit web application**.

---
*Report compiled by Bankim Kamila | ITM Skills University B.Tech CSE (2024–28)*
