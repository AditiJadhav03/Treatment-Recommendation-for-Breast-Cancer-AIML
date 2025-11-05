import pandas as pd
import numpy as np

# --- Total samples ---
N = 2100   # 700 per treatment

def random_choice(options):
    return np.random.choice(options)

records = []

# -------------------------------------------
# Create balanced & clinically distinct data
# -------------------------------------------
for i in range(N):

    if i < N/3:
        # ------------------ CHEMOTHERAPY ------------------
        treatment = 'Chemotherapy'
        age = np.random.randint(35, 75)
        lymph_nodes_examined = np.random.randint(8, 25)
        tumor_nuclei_percent = np.random.randint(75, 95)
        er_status = random_choice(['negative', 'negative', 'positive'])
        pr_status = random_choice(['negative', 'negative', 'positive'])
        her2_status = random_choice(['positive', 'positive', 'negative'])
        menopause_status = random_choice(['post', 'post', 'pre'])
        surgery_type = 'mastectomy'
        histology_type = random_choice([
            'infiltrating ductal carcinoma',
            'infiltrating lobular carcinoma'
        ])
        pathologic_stage = random_choice(['stage iii', 'stage iv'])
        pathologic_t = random_choice(['t3', 't4'])
        pathologic_n = random_choice(['n2', 'n3'])
        pathologic_m = random_choice(['m0', 'm1'])
        tumor_necrosis = random_choice(['partial', 'yes'])
        anatomic_subdivision = random_choice([
            'left upper outer', 'right upper outer',
            'left lower inner', 'right lower inner'
        ])

    elif i < 2*N/3:
        # ------------------ HORMONAL THERAPY ------------------
        treatment = 'Hormonal_Therapy'
        age = np.random.randint(45, 70)
        lymph_nodes_examined = np.random.randint(2, 10)
        tumor_nuclei_percent = np.random.randint(45, 65)
        er_status = 'positive'
        pr_status = 'positive'
        her2_status = 'negative'
        menopause_status = random_choice(['pre', 'post'])
        surgery_type = random_choice(['lumpectomy', 'mastectomy'])
        histology_type = random_choice([
            'infiltrating ductal carcinoma',
            'infiltrating lobular carcinoma'
        ])
        pathologic_stage = random_choice(['stage ii'])
        pathologic_t = random_choice(['t1', 't2'])
        pathologic_n = random_choice(['n0', 'n1'])
        pathologic_m = 'm0'
        tumor_necrosis = random_choice(['no', 'partial'])
        anatomic_subdivision = random_choice([
            'left upper outer', 'right upper outer',
            'left lower inner', 'right lower inner'
        ])

    else:
        # ------------------ SURGERY ------------------
        treatment = 'Surgery'
        age = np.random.randint(25, 60)
        lymph_nodes_examined = np.random.randint(0, 4)
        tumor_nuclei_percent = np.random.randint(10, 40)
        er_status = random_choice(['positive', 'negative'])
        pr_status = random_choice(['positive', 'negative'])
        her2_status = 'negative'
        menopause_status = random_choice(['pre', 'pre', 'post'])
        surgery_type = 'lumpectomy'
        histology_type = random_choice([
            'infiltrating ductal carcinoma',
            'infiltrating lobular carcinoma'
        ])
        pathologic_stage = 'stage i'
        pathologic_t = 't1'
        pathologic_n = 'n0'
        pathologic_m = 'm0'
        tumor_necrosis = 'no'
        anatomic_subdivision = random_choice([
            'left upper outer', 'right upper outer',
            'left lower inner', 'right lower inner'
        ])

    records.append([
        age, lymph_nodes_examined, tumor_nuclei_percent,
        er_status, pr_status, her2_status, menopause_status,
        surgery_type, histology_type, pathologic_stage,
        pathologic_t, pathologic_n, pathologic_m,
        tumor_necrosis, anatomic_subdivision, treatment
    ])

# -------------------------------------------------------
# Create DataFrame & save balanced dataset
# -------------------------------------------------------
cols = [
    'age', 'lymph_nodes_examined', 'tumor_nuclei_percent',
    'er_status', 'pr_status', 'her2_status', 'menopause_status',
    'surgery_type', 'histology_type', 'pathologic_stage',
    'pathologic_t', 'pathologic_n', 'pathologic_m',
    'tumor_necrosis', 'anatomic_subdivision', 'treatment'
]

df = pd.DataFrame(records, columns=cols)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

output_path = "data/balanced_bc_clinical_2000.csv"
df.to_csv(output_path, index=False)

print(f" Balanced dataset with realistic patterns saved to {output_path}")
print(df['treatment'].value_counts())

