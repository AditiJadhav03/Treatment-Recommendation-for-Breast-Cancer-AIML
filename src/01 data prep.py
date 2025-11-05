import pandas as pd
import numpy as np
import os

# 1. FILE PATHS ---
RAW_FILE_PATH = "data/balanced_bc_clinical_2000.csv"
PROCESSED_FILE_PATH = "data/processed_balanced_bc_clinical_2000.csv"


# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# --- 2. DATA LOADING & CLEANING ---
print(f"Loading new data from: {RAW_FILE_PATH}")
df = pd.read_csv(RAW_FILE_PATH, low_memory=False)

df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.lower()

# Standardize all text/object columns (except the target)
text_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in text_cols:
    df.loc[:, col] = df[col].astype(str).str.lower().str.strip()
    # Replace common missing values with 'not_reported'
    df.loc[:, col] = df[col].replace(['nan', 'unknown', 'no_info', '--', 'unspecified'], 'not_reported')
    
# --- 3. TARGET CLEANING & GROUPING (CRITICAL) ---

TARGET_COL = 'treatment'

if TARGET_COL not in df.columns:
    print(" FATAL ERROR: The 'Treatment' column is missing from the new CSV.")
    exit()

# Grouping logic to ensure only the 3 desired classes remain, plus 'Other'
def map_treatment(x):
    if 'chemo' in x:
        return 'Chemotherapy'
    elif 'hormonal' in x:
        return 'Hormonal_Therapy'
    elif 'surgery' in x or 'lumpectomy' in x or 'mastectomy' in x:
        return 'Surgery'
    else:
        return 'Other'

df.loc[:, TARGET_COL] = df[TARGET_COL].apply(map_treatment)

print("\n Target Classes Created:")
print(df[TARGET_COL].value_counts())
print("-" * 30)

# --- 4. FEATURE TRANSFORMATION ---

# Binary/Ordinal Features (direct mapping)
df['er_status'] = df['er_status'].map({'positive': 1, 'negative': 0, 'not_reported': 0}).fillna(0)
df['pr_status'] = df['pr_status'].map({'positive': 1, 'negative': 0, 'not_reported': 0}).fillna(0)
df# HER2_Status: map positive to 1, everything else to 0
df['her2_status'] = df['her2_status'].apply(lambda x: 1 if x in ['positive', 'equivocal', 'indeterminate'] else 0)

# Numeric Features (ensure they are numeric)
numeric_cols = ['age', 'lymph_nodes_examined', 'tumor_nuclei_percent']
for col in numeric_cols:
    df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

# --- 5. FINAL ONE-HOT ENCODING ---

# Identify remaining categorical columns (excluding the now-numeric/binary ones)
nominal_cols = [
    'menopause_status', 'surgery_type', 'histology_type', 
    'pathologic_stage', 'pathologic_t', 'pathologic_n', 
    'pathologic_m', 'tumor_necrosis', 'anatomic_subdivision', TARGET_COL
]

df = pd.get_dummies(df, columns=nominal_cols, drop_first=False, prefix_sep='__')

# Debug check: confirm all three classes exist
treatment_cols = [c for c in df.columns if c.startswith('treatment__')]
print("\nTreatment columns created:", treatment_cols)


# --- 6. FINAL SAVE ---
df.to_csv(PROCESSED_FILE_PATH, index=False, encoding='latin1') 

print(f"\n Data preparation complete!")
print(f"The model-ready file has been saved to: {PROCESSED_FILE_PATH}")
print(f"Final dataset shape: {df.shape}")