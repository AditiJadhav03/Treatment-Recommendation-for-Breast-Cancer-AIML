import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = "models/random_forest_multiclass_final_model.pkl"
FEATURES_PATH = "models/multiclass_final_feature_names.pkl"

# The three classes the model predicts
TARGET_CLASSES = ['Chemotherapy', 'Hormonal_Therapy', 'Surgery']

# --- Load Model and Features ---
@st.cache_resource
def load_model_assets():
    """Loads the trained model and feature names."""
    try:
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        return model, feature_names
    except FileNotFoundError:
        st.error(f"Model files not found. Ensure '{MODEL_PATH}' and '{FEATURES_PATH}' exist after running 02.model train.py.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        st.stop()

model, feature_names = load_model_assets()


# --- Feature Mapping Dictionaries (Based on your 01.data prep.py logic) ---
FEATURE_OPTIONS = {
    # Binary Features
    'er_status': {'Positive': 1, 'Negative': 0},
    'pr_status': {'Positive': 1, 'Negative': 0},
    'her2_status': {'Positive/Indeterminate/Equivocal': 1, 'Negative': 0},

    # Numeric Features
    'age': 55,
    'lymph_nodes_examined': 10,
    'tumor_nuclei_percent': 70,

    # Nominal Features
    'menopause_status': ['post', 'pre', 'unknown'],
    'surgery_type': ['mastectomy', 'lumpectomy', 'no_surgery/other'],
    'histology_type': ['infiltrating ductal carcinoma', 'infiltrating lobular carcinoma', 'other'],
    'pathologic_stage': ['stage i', 'stage ii', 'stage iii', 'stage iv', 'stage 0'],
    'pathologic_t': ['t1', 't2', 't3', 't4'],
    'pathologic_n': ['n0', 'n1', 'n2', 'n3'],
    'pathologic_m': ['m0', 'm1'],
    'tumor_necrosis': ['no', 'partial', 'yes'],
    'anatomic_subdivision': ['left upper outer', 'right upper outer', 'left lower inner', 'right lower inner', 'other']
}


# --- Prediction Function (UPDATED & FIXED) ---
def make_prediction(input_data: dict, feature_names: list, model):
    # --- NEW: Normalize categorical values before encoding ---
    for key in list(input_data.keys()):
        val = str(input_data[key]).strip().lower()
        val = val.replace(' ', '_').replace('/', '_')
        input_data[key] = val

    # 1. Create final feature vector
    final_features_df = pd.DataFrame(0, index=[0], columns=feature_names)

    # Numeric features
    NUMERIC_FEATURES = ['age', 'lymph_nodes_examined', 'tumor_nuclei_percent']
    for feature in NUMERIC_FEATURES:
        if feature in input_data:
            final_features_df.loc[0, feature] = input_data[feature]

    # 2. Process all categorical/OHE features
    for col_base, selected_value in input_data.items():
        if col_base in NUMERIC_FEATURES:
            continue

        selected_value_standardized = str(selected_value).lower().replace(' ', '_').replace('/', '_')
        ohe_col_name = f"{col_base}__{selected_value_standardized}"

        if ohe_col_name in feature_names:
            final_features_df.loc[0, ohe_col_name] = 1

    # 3. Predict probabilities
    final_features_df = final_features_df.astype('float64')
    proba_raw = model.predict_proba(final_features_df)[0]

    # 4. Build results
    model_classes_encoded = model.classes_
    results = {}
    for i, class_label_encoded in enumerate(model_classes_encoded):
        class_name = class_label_encoded.split('__')[-1].replace('_', ' ').title()
        results[class_name] = round(proba_raw[i] * 100, 2)

    # Handle missing baseline class (Chemotherapy)
    hormonal_prob = results.get('Hormonal Therapy', 0.00)
    surgery_prob = results.get('Surgery', 0.00)
    chemo_prob = round(100.0 - hormonal_prob - surgery_prob, 2)
    final_probabilities = {
        'Chemotherapy': max(0.00, chemo_prob),
        'Hormonal Therapy': hormonal_prob,
        'Surgery': surgery_prob
    }

    best_treatment = max(final_probabilities, key=final_probabilities.get)
    sorted_probs = dict(sorted(final_probabilities.items(), key=lambda item: item[1], reverse=True))

    return {
        "Predicted_Treatment": best_treatment,
        "Probability_Score": sorted_probs[best_treatment],
        "All_Probabilities": sorted_probs
    }


# --- Streamlit UI ---
st.set_page_config(page_title="BC Treatment Predictor", layout="wide")

st.title("Breast Cancer Treatment Classification")
st.markdown("Use the controls below to input patient data and predict the most likely treatment (Chemotherapy, Hormonal Therapy, or Surgery).")

with st.sidebar:
    st.header("Patient Clinical Data")

    # Numeric Inputs
    age = st.slider("Age", 20, 90, FEATURE_OPTIONS['age'])
    lymph_nodes_examined = st.slider("Lymph Nodes Examined", 0, 40, FEATURE_OPTIONS['lymph_nodes_examined'])
    tumor_nuclei_percent = st.slider("Tumor Nuclei Percent", 10, 100, FEATURE_OPTIONS['tumor_nuclei_percent'])

    st.markdown("---")
    st.subheader("Hormone Receptor Status")
    er_status_key = st.radio("ER Status", list(FEATURE_OPTIONS['er_status'].keys()))
    pr_status_key = st.radio("PR Status", list(FEATURE_OPTIONS['pr_status'].keys()))
    her2_status_key = st.radio("HER2 Status", list(FEATURE_OPTIONS['her2_status'].keys()))

    st.markdown("---")
    st.subheader("Disease Characteristics")
    menopause_status = st.selectbox("Menopause Status", FEATURE_OPTIONS['menopause_status'])
    surgery_type = st.selectbox("Surgery Type", FEATURE_OPTIONS['surgery_type'])
    histology_type = st.selectbox("Histology Type", FEATURE_OPTIONS['histology_type'])
    pathologic_stage = st.selectbox("Pathologic Stage", FEATURE_OPTIONS['pathologic_stage'])
    pathologic_t = st.selectbox("Pathologic T", FEATURE_OPTIONS['pathologic_t'])
    pathologic_n = st.selectbox("Pathologic N", FEATURE_OPTIONS['pathologic_n'])
    pathologic_m = st.selectbox("Pathologic M", FEATURE_OPTIONS['pathologic_m'])
    tumor_necrosis = st.selectbox("Tumor Necrosis", FEATURE_OPTIONS['tumor_necrosis'])
    anatomic_subdivision = st.selectbox("Anatomic Subdivision", FEATURE_OPTIONS['anatomic_subdivision'])

# Assemble user input
input_data = {
    'age': age,
    'lymph_nodes_examined': lymph_nodes_examined,
    'tumor_nuclei_percent': tumor_nuclei_percent,
    'er_status': er_status_key,
    'pr_status': pr_status_key,
    'her2_status': her2_status_key,
    'menopause_status': menopause_status,
    'surgery_type': surgery_type,
    'histology_type': histology_type,
    'pathologic_stage': pathologic_stage,
    'pathologic_t': pathologic_t,
    'pathologic_n': pathologic_n,
    'pathologic_m': pathologic_m,
    'tumor_necrosis': tumor_necrosis,
    'anatomic_subdivision': anatomic_subdivision
}

# Prediction button
if st.button("Predict Treatment"):
    st.subheader("Prediction Results")

    prediction_result = make_prediction(input_data, feature_names, model)

    best_treatment = prediction_result['Predicted_Treatment']
    best_score = prediction_result['Probability_Score']

    if best_score > 70:
        st.success(f"Best Suggested Treatment: **{best_treatment}**")
        st.subheader(f"Confidence: {best_score:.2f}%")
    else:
        st.info(f"Primary Suggested Treatment: **{best_treatment}**")
        st.subheader(f"Confidence: {best_score:.2f}%")

    st.markdown("---")
    st.subheader("Probability Breakdown (All Options)")

    col1, col2, col3 = st.columns(3)
    for i, (treatment, prob) in enumerate(prediction_result['All_Probabilities'].items()):
        col = [col1, col2, col3][i % 3]
        with col:
            st.metric(label=f"Probability for {treatment}", value=f"{prob:.2f}%")
