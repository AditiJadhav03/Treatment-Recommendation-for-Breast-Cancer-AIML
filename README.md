# Treatment Recommendation for Breast Cancer AIML

## Project Overview
Breast cancer is one of the leading causes of cancer-related deaths among women worldwide. Selecting the most effective treatment for a patient can be challenging due to the complexity and variability of the disease. This project leverages **Artificial Intelligence (AI) and Machine Learning (ML)** to recommend personalized treatment options for breast cancer patients based on their clinical data.

The system analyzes patient data, processes features, trains predictive models, and provides treatment recommendations through a simple interface.

---

## Problem Statement
Traditional breast cancer treatment decisions rely heavily on physician expertise and generalized protocols. This can lead to:
- Over-treatment or under-treatment
- Longer decision-making time
- Variability in treatment outcomes

**Goal:** Build a machine learning-based system that predicts the most suitable treatment for a patient, improving efficiency and accuracy in clinical decision-making.

---

## Dataset
The project uses clinical datasets containing patient information such as:
- Age
- Tumor characteristics
- Biomarkers
- Histopathological features
- Treatment outcomes

**Files in this project:**
- `data/balanced_bc_clinical_2000.csv` – Balanced dataset for training
- `data/processed_balanced_bc_clinical_2000.csv` – Preprocessed dataset with selected features
- `data/bc_treatment_dataset.csv` – Original dataset (optional)

> All sensitive patient data has been anonymized.

---

## How It Works
1. **Data Preprocessing (`src/01 data prep.py`)**
   - Cleans the raw data
   - Handles missing values
   - Balances the dataset to reduce bias
   - Extracts relevant features for model training

2. **Model Training (`src/02 model train.py`)**
   - Trains multiple machine learning models
   - Performs feature selection and optimization
   - Saves the trained model and feature names for inference

3. **Model Deployment (`src/04_streamlit_demo.py`)**
   - Uses **Streamlit** to create an interactive interface
   - Allows users to input patient data
   - Predicts and recommends the most suitable treatment option

---

## Technologies Used
- **Programming Language:** Python  
- **Libraries:** 
  - `pandas`, `numpy` – Data processing
  - `scikit-learn` – Machine learning models
  - `pickle` – Save/load trained models
  - `streamlit` – Web app interface
- **Version Control:** Git & GitHub  
- **Environment:** Virtual environment (`venv`)  

---

## Output
The project outputs:
1. A **trained predictive model** capable of recommending breast cancer treatments.
2. **Interactive web app** to input patient data and get treatment suggestions.
3. **Model performance metrics** (accuracy, precision, recall) to evaluate predictions.
4. **Processed datasets** and feature selections for reproducibility.



