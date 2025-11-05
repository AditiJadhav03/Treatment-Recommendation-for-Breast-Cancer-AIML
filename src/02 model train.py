import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os
import re
from imblearn.over_sampling import SMOTE  # <-- NEW IMPORT

# --- 1. CONFIGURATION ---
PROCESSED_FILE_PATH = "data/processed_features_multiclass_final.csv"
MODEL_SAVE_PATH = "models/random_forest_multiclass_final_model.pkl"
FEATURES_SAVE_PATH = 'models/multiclass_final_feature_names.pkl'

os.makedirs('models', exist_ok=True)
TARGET_SEARCH_PATTERN = r'treatment__' # Matches the prefix from one-hot encoding

# --- 2. DATA LOADING & CLEANING ---
print(f"Loading processed features from: {PROCESSED_FILE_PATH}")
try:
    df = pd.read_csv(PROCESSED_FILE_PATH, encoding='latin1')
except FileNotFoundError:
    print(f" ERROR: {PROCESSED_FILE_PATH} not found. Please run 01.data prep.py first.")
    exit()

df.columns = df.columns.str.lower()
    
# --- 3. TARGET SELECTION (MULTI-CLASS METHOD) ---

# Find ALL one-hot encoded treatment columns 
target_cols_encoded = [col for col in df.columns if re.search(TARGET_SEARCH_PATTERN, col)]

if len(target_cols_encoded) < 1:
    print(" FATAL ERROR: Multi-class target must have at least 3 encoded treatment columns (Chemo, Hormonal, Surgery).")
    print("Found columns:", target_cols_encoded)
    exit()

# Features (X) are everything not in the target
X = df.drop(columns=target_cols_encoded)
# Target (Y) is the set of all one-hot encoded columns
Y = df[target_cols_encoded] 

print(f"Training multi-class model with {len(Y.columns)} target classes.")
print(f"Target Classes (excluding baseline): {[c.split('__')[1] for c in Y.columns.tolist()]}")

# Convert one-hot to a single label column for training
Y_labels = Y.idxmax(axis=1)

# --- 4. DATA SPLIT ---
X_train, X_test, Y_train_labels, Y_test_labels = train_test_split(
    X, Y_labels, 
    test_size=0.2, 
    random_state=42, 
    # Stratify is crucial for splitting data evenly based on the initial labels
    stratify=Y_labels
)

# --- 5. OVERSAMPLING (The FIX for Bias) ---
print("\nApplying SMOTE to aggressively balance training data...")
smote = SMOTE(random_state=42)
X_train_resampled, Y_train_labels_resampled = smote.fit_resample(X_train, Y_train_labels)

print(f"Original Training set size: {len(X_train)} cases")
print(f"Resampled Training set size: {len(X_train_resampled)} cases (Now perfectly balanced)")


# --- 6. MODEL TRAINING ---
print("\nTraining Multi-Class Random Forest Classifier...")

# The model now trains on the perfectly balanced, resampled data
model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42,
    # class_weight is technically not strictly necessary after SMOTE, but harmless
    class_weight='balanced', 
    max_features='sqrt',
    min_samples_leaf=5 
)

model.fit(X_train_resampled, Y_train_labels_resampled)

print(" Multi-class model training complete.")

# --- 7. EVALUATION ---
# Evaluation must be done on the original, non-resampled test set (X_test)
Y_pred_labels = model.predict(X_test)

print("\n--- Model Performance on Test Set (Multi-Class) ---")
# Extract simple class names for the report
class_names = [c.split('__')[1] for c in Y_labels.unique()]
print(classification_report(Y_test_labels, Y_pred_labels, target_names=class_names, zero_division=0)) 

# --- 8. MODEL SAVING ---
joblib.dump(model, MODEL_SAVE_PATH)
joblib.dump(X.columns.tolist(), FEATURES_SAVE_PATH) 

print(f"\n Multi-class model saved successfully to: {MODEL_SAVE_PATH}")
print("Ready to move to 03.flask api.py for deployment.")