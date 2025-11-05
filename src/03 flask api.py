import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import os

# --- 1. CONFIGURATION ---
MODEL_PATH = 'models/random_forest_multiclass_final_model.pkl'
FEATURES_PATH = 'models/multiclass_final_feature_names.pkl'
# The list of possible treatment outputs (must match the classes created by 01.data prep.py)
TARGET_CLASSES = ['Chemotherapy', 'Hormonal_Therapy', 'Surgery', 'Other']

# --- 2. LOAD ARTIFACTS ---
try:
    # Load the trained model and features
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print("Model and features loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model artifacts: {e}. Ensure 02.model train.py was run successfully.")
    exit()

# --- 3. FLASK SETUP ---
app = Flask(__name__)

# --- 4. PREPROCESSING & PREDICTION ---

def preprocess_and_predict(raw_data):
    """
    Takes raw patient data (as a dict), transforms it to the model's feature format, and predicts.
    """
    
    # 1. Create a DataFrame from the input data (single row)
    input_df = pd.DataFrame([raw_data])
    
    # Standardize column names
    input_df.columns = input_df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Standardize all incoming text values
    for col in input_df.select_dtypes(include=['object']).columns:
        input_df[col] = input_df[col].astype(str).str.lower().str.strip()
        
    # Apply the same binary/ordinal mappings as in 01.data prep.py
    input_df['er_status'] = input_df['er_status'].map({'positive': 1, 'negative': 0}).fillna(0)
    input_df['pr_status'] = input_df['pr_status'].map({'positive': 1, 'negative': 0}).fillna(0)
    input_df['her2_status'] = input_df['her2_status'].apply(lambda x: 1 if x in ['positive', 'equivocal', 'indeterminate'] else 0)

    # Ensure numeric columns are present and numeric
    numeric_cols = ['age', 'lymph_nodes_examined', 'tumor_nuclei_percent']
    for col in numeric_cols:
        if col not in input_df.columns:
            input_df[col] = 0 
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0) 

    # 2. One-Hot Encode Categorical features
    
    nominal_cols = [
        'menopause_status', 'surgery_type', 'histology_type', 
        'pathologic_stage', 'pathologic_t', 'pathologic_n', 
        'pathologic_m', 'tumor_necrosis', 'anatomic_subdivision'
    ]
    
    # Drop columns that aren't features or target
    input_df = input_df.drop(columns=['treatment'], errors='ignore')
    
    # Add prefix for encoding
    input_df = pd.get_dummies(input_df, columns=nominal_cols, drop_first=True, prefix_sep='__')
    
    # 3. Align Features (CRUCIAL STEP)
    # Create a DataFrame of zeros for all expected features (from the training set)
    final_features = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Fill in the features present in the input
    for col in input_df.columns:
        if col in final_features.columns:
            final_features[col] = input_df[col].iloc[0]

    # 4. Predict probabilities
    prediction_prob_raw = model.predict_proba(final_features)[0]
    
    # The model predicts the probability for each class (Chemo, Hormonal, Surgery, Other)
    # We must match the probability output order to the sorted class names used in training.
    
    # 5. Structure Output (FINAL FIX: Ensure all 3 classes are accounted for)
    
    # Get the class labels from the model (e.g., 'treatment__Chemotherapy')
    model_classes = model.classes_ 
    
    results = {}
    total_encoded_prob = 0.0
    
    # Map the probabilities from the raw output array to the class names
    for i, class_label_encoded in enumerate(model_classes):
        class_name = class_label_encoded.split('__')[-1] # e.g., 'Chemotherapy'
        prob = round(prediction_prob_raw[i] * 100, 2)
        results[class_name] = prob
        total_encoded_prob += prob

    # The issue: If the classes are Hormonal and Surgery, Chemotherapy is calculated by subtraction.
    # We must explicitly add the Chemotherapy class if it's missing (it's the baseline).
    # Since the model classes are generally sorted alphabetically, let's use the known 3 classes:
    
    # The three known classes we want in the final output
    final_output_classes = ['Chemotherapy', 'Hormonal_Therapy', 'Surgery']

    # --- FINAL PROBABILITY ASSEMBLY ---
    
    # Use a dictionary to hold the three final outputs
    final_probabilities = {}
    
    # Ensure Hormonal and Surgery (the encoded columns) are pulled from results
    final_probabilities['Hormonal_Therapy'] = results.get('Hormonal_Therapy', 0.00)
    final_probabilities['Surgery'] = results.get('Surgery', 0.00)

    # Calculate Chemotherapy (The Baseline)
    # The sum of all probabilities must be 100.
    chemo_prob = round(100.0 - final_probabilities['Hormonal_Therapy'] - final_probabilities['Surgery'], 2)
    
    # Ensure no negative zero due to floating point math
    final_probabilities['Chemotherapy'] = max(0.00, chemo_prob) 
    
    # Identify the best prediction from the final set
    best_treatment = max(final_probabilities, key=final_probabilities.get)
    
    # Sort for cleaner display
    sorted_probs = dict(sorted(final_probabilities.items(), key=lambda item: item[1], reverse=True))

    return {
        "Predicted_Treatment": best_treatment,
        "Probability_Score": sorted_probs[best_treatment],
        "All_Probabilities": sorted_probs
    }

# --- 5. API ENDPOINT ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON data from the user and returns a prediction.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided. Send JSON with patient features."}), 400
        
        prediction_result = preprocess_and_predict(data)
        
        return jsonify(prediction_result)
        
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

# --- 6. RUN THE APP ---
if __name__ == '__main__':
    # To run the app, use: python 03.flask api.py
    # Then access http://127.0.0.1:5000/predict with a POST request.
    app.run(debug=True, port=5000)