# Visual Evidence & Report Images Catalog

**Project Title:** Restaurant Review Sentiment Analysis Using Machine Learning  
**Case Study:** Case Study No. 77 – Sentiment Analysis  
**Institution:** ITM Skills University | B.Tech CSE (2024–28)  

This directory contains the high-resolution (300 DPI) visual evidence images generated directly from actual dataset records and machine learning experiment results for the college project report.

---

## Visual Evidence Table

| Image Number | File Name | Purpose / Content Description | Where to Use in Report |
|---|---|---|---|
| **01** | `01_raw_dataset.png` | Displays first 10 authentic records from UCI Yelp dataset before preprocessing (`Review`, `Liked`, `Sentiment`). | **Dataset & Data Collection Chapter** |
| **02** | `02_dataset_information.png` | Comprehensive metadata card showing row count (1,000), columns (3), data types, missing values (0), and duplicate checks. | **Dataset Overview & Technical Specifications Chapter** |
| **03** | `03_data_cleaning.png` | Before vs After cleaning table showing duplicate removal (-4 rows -> 996 rows) and text transformation examples. | **Data Cleaning & Quality Verification Chapter** |
| **04** | `04_sentiment_distribution.png` | Bar chart and pie proportion of positive (499 / 50.1%) vs negative (497 / 49.9%) sentiment classes. | **Exploratory Data Analysis (EDA) Chapter** |
| **05** | `05_rating_distribution.png` | Binary user satisfaction (`Liked` column) and 1-to-5 star rating equivalent distribution. | **Exploratory Data Analysis (EDA) Chapter** |
| **06** | `06_review_length.png` | Dual histogram & KDE distribution analysis for review word count and character length. | **Exploratory Data Analysis (EDA) Chapter** |
| **07** | `07_text_preprocessing.png` | Before vs After text normalization table with 6 genuine customer reviews (lowercased, special chars stripped). | **Text Preprocessing & NLP Pipeline Chapter** |
| **08** | `08_tfidf_features.png` | Sample TF-IDF numerical feature matrix showing non-zero token weights across top restaurant vocabulary words. | **Feature Engineering & TF-IDF Chapter** |
| **09** | `09_train_test_split.png` | Architecture diagram of 80% Training (796 rows) vs 20% Unseen Testing (200 rows) with stratified split. | **Experimental Setup & Dataset Partitioning Chapter** |
| **10** | `10_model_comparison.png` | Grouped bar chart comparing actual test Accuracy, Precision, Recall, and F1-Score for Logistic Regression (83.00%) vs Decision Tree (71.50%). | **Model Training & Comparative Evaluation Chapter** |
| **11** | `11_accuracy_summary.png` | Summary comparison card table detailing all performance metrics and rationale for model selection. | **Model Performance Summary Chapter** |
| **12** | `12_confusion_matrix.png` | High-resolution Confusion Matrix for Logistic Regression (TN: 81, FP: 19, FN: 15, TP: 85) on 200 test samples. | **Model Evaluation & Error Matrix Chapter** |
| **13** | `13_classification_report.png` | Full Classification Report table with precision, recall, f1-score, and support for Negative, Positive, and Macro/Weighted averages. | **Classification Metrics & Evaluation Chapter** |
| **14** | `14_error_analysis.png` | Detailed error analysis table analyzing authentic misclassified test reviews, reasons for misclassification, and NLP root causes. | **Error Analysis & Limitations Chapter** |
| **15** | `15_best_model.png` | Final decision card highlighting the selected best model (Logistic Regression, 83.00% Accuracy, 81.73% Precision) and technical rationale. | **Final Model Selection & Architecture Chapter** |
| **16** | `16_streamlit_home.png` | High-fidelity Streamlit web application screenshot showing the home page, quick preset buttons, and review text input area. | **Web Application & UI Deployment Chapter** |
| **17** | `17_streamlit_positive.png` | Streamlit web application screenshot demonstrating real-time positive sentiment inference with confidence probabilities. | **Web Application & UI Deployment Chapter** |
| **18** | `18_streamlit_negative.png` | Streamlit web application screenshot demonstrating real-time negative sentiment inference with confidence probabilities. | **Web Application & UI Deployment Chapter** |
| **19** | `19_streamlit_neutral.png` | Streamlit web application screenshot demonstrating neutral/mixed sentiment classification and probability breakdown. | **Web Application & UI Deployment Chapter** |
| **20** | `20_project_workflow.png` | Complete end-to-end architecture diagram featuring branching parallel models (Logistic Regression vs Decision Tree) through evaluation and Streamlit deployment. | **Methodology & System Architecture Chapter** |

---

## Instructions for Report Compilation

1. **Format:** All PNG images are rendered at 300 DPI with light backgrounds, modern sans-serif typography, and standard publication dimensions (16:9 / balanced ratios).
2. **Embedding:** Insert these PNG figures directly into your LaTeX, Microsoft Word, or Google Docs report document under their respective chapters.
3. **Reproducibility:** To regenerate all figures with updated parameters or styles, run:
   ```bash
   python3 generate_visualizations.py
   ```
