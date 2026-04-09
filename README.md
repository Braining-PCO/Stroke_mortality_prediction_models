# Stroke Mortality Prediction Models
### A multi-horizon mortality prediction framework using MIMIC-IV clinical data

---

## Overview

This project develops and evaluates machine learning and deep learning models for predicting **stroke mortality** across multiple time horizons (1, 15, 30, 45, 60, and 90 days post-admission) using the [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) clinical database. The pipeline integrates structured EHR data with information extracted from free-text clinical notes via Named Entity Recognition (NER), and provides explainability using SHAP and LIME.

---

## Pipeline Overview

```
MIMIC-IV Raw Data
      │
      ▼
┌─────────────────────────┐
│  Exploratory Data       │  ── eda_notebook/
│  Analysis (EDA)         │
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│  Data Preprocessing     │  ── preprocessing_notebooks/
│  • Clinical note        │
│    selection            │
│  • Vital signs          │
│  • Baseline features    │
│  • NER extraction       │
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│  Model Training         │  ── models_notebooks/
│  • ML: XGBoost          │
│  • DL: Multi-task NN    │
│  Horizons: 1, 15, 30,   │
│  45, 60, 90 days        │
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│  Explainability         │  ── models_notebooks/
│  • SHAP                 │
│  • LIME                 │
│  • Integrated Gradients │
└─────────────────────────┘
```

---

## Repository Structure

```
stroke-mortality-prediction/
│
├── README.md
│
├── eda_notebook/
│   ├── eda.ipynb 
│   │
│── eda_python_file/
│   │── eda.py       
│   │ 
│── models_notebooks/          
│   │__ dl_model.ipynb
|   |__ ml_model.ipynb
│   │
│── models_python_files/
│   |── dl_model.py               
│   |── ml_model.py              
│   |
├── preprocessing_notebooks/
│   ├── add_clinical_notes.ipynb
│   |── add_vital_signs.ipynb
│   |── create_baseline_data.ipynb
|   |── ner_extraction.ipynb
│   |
└── preprocessing_python_files/
│   ├── add_clinical_notes.py
│   └── add_vital_signs.py
│   └── create_baseline_data.py
|   └── ner_extraction.py
```

---

## Stage 1 — Exploratory Data Analysis

**Notebook:** [`eda_notebook/eda.ipynb`](eda_notebook/eda.ipynb)

The EDA notebook investigates the MIMIC-IV stroke cohort to understand the data distributions, class imbalances, and clinical patterns before any modelling takes place. Key analyses include:

- **Cohort description** — demographics (age group, gender, race), stroke type distribution (hemorrhagic vs. ischemic), and marital status
- **Mortality statistics** — overall in-hospital mortality rate and temporal distribution of deaths across the six prediction horizons
- **Missing data audit** — null counts and imputation strategy decisions for vitals and structured features
- **Correlation analysis** — pairwise feature correlation heatmap to identify multicollinearity and inform feature selection
- **Class imbalance profiling** — mortality event counts per horizon, guiding the use of `scale_pos_weight` in XGBoost and class-weighted losses in the neural network

---

## Stage 2 — Data Preprocessing

### 2.1 Clinical Note Selection

**Notebook:** [`preprocessing_notebooks/add_clinical_notes.ipynb`](preprocessing_notebooks/add_clinical_notes.ipynb)

Not all notes in MIMIC-IV are clinically relevant to stroke outcome prediction. This notebook filters the free-text notes corpus to retain only notes that are:

- Associated with a confirmed stroke diagnosis (ICD codes for ischemic and hemorrhagic stroke)
- Written within the admission window of interest

### 2.2 Vital Signs Integration

**Notebook:** [`preprocessing_notebooks/add_vital_signs.ipynb`](preprocessing_notebooks/add_vital_signs.ipynb)

Blood pressure values from the `chartevents` table are extracted, cleaned, and aggregated per admission. Values are classified as **normal** or **abnormal** based on clinical thresholds and encoded as a binary variable for model input.

### 2.3 Baseline Data Construction

**Notebook:** [`preprocessing_notebooks/create_baseline_data.ipynb`](preprocessing_notebooks/create_baseline_data.ipynb)

Structured baseline features are assembled and encoded for model training:

| Feature | Encoding |
|---|---|
| Gender | One-hot (`Sex_F`, `Sex_M`) |
| Race | Binary (`White` / `Non-White`) then one-hot |
| Stroke class | One-hot (`sc_Hemorrhagic`, `sc_Ischemic`) |
| Marital status → Care level | Ordinal map: `MARRIED=high`, `SINGLE/DIVORCED=medium`, `WIDOWED=low` |
| Readmissions | Binary (any prior readmission: yes/no) |
| Blood pressure | Binary (normal=0, abnormal=1) |
| Age group | Ordinal (`18–44=1` → `85+=5`) |

The **target** for each horizon `h` is a binary label: did this patient die within `h` days of admission? Patients discharged or censored before day `h` are excluded from that horizon's training set to avoid look-ahead bias.

### 2.4 NER Extraction

**Notebook:** [`preprocessing_notebooks/ner_extraction.ipynb`](preprocessing_notebooks/ner_extraction.ipynb)

Named Entity Recognition is applied to the selected clinical notes using to extract clinical entities. The 12 entity classes extracted are:

`DOSAGE` · `MEDICATION` · `SIGN_SYMPTOM` · `DISEASE_DISORDER` · `BIOLOGICAL_STRUCTURE` · `LAB_VALUE` · `DIAGNOSTIC_PROCEDURE` · `SEVERITY` · `THERAPEUTIC_PROCEDURE` · `CLINICAL_EVENT` · `DETAILED_DESCRIPTION` · `DURATION`

Extracted entities are encoded into features via a two-step pipeline:

1. **Canonical entity selection** — entities within each class are clustered using agglomerative clustering on Bio_ClinicalBERT embeddings (distance threshold = 0.25) to identify the top-K representative canonical terms
2. **Feature encoding** — each patient's entity list is encoded as class-normalized counts and binary presence flags for the selected canonical entities

---

## Stage 3 — Model Training

### 3.1 Machine Learning Model — XGBoost

**Notebook:** [`models_notebooks/ml_model.ipynb`](models_notebooks/ml_model.ipynb)

An **XGBoost** gradient boosted classifier is trained using an expanded multi-horizon dataset. All six horizons are stacked into a single training set with `horizon` added as an explicit feature, allowing the single model to learn across the full temporal range.

**Key training details:**
- `objective`: `binary:logistic`
- `max_depth`: 5, `learning_rate`: 0.05, `n_estimators`: 200
- Class imbalance handled via `scale_pos_weight`
- 80/20 patient-level train/test split (no patient appears in both sets)
- Monotonicity post-processing: predicted mortality probabilities are enforced to be non-decreasing over time for each patient

**Evaluation:** AUC-ROC, accuracy, and confusion matrices are reported per horizon. A single-patient risk trajectory can be visualised across all six horizons.

**Explainability (ML):**
- **SHAP** — global feature importance and per-patient waterfall plots
- **LIME** — local perturbation-based explanations for individual predictions
- Permutation-based horizon-specific feature importance

### 3.2 Deep Learning Model — Multi-Task Neural Network

**Notebook:** [`models_notebooks/dl_model.ipynb`](models_notebooks/dl_model.ipynb)

A **multi-task neural network** built with TensorFlow/Keras is trained jointly across all six horizons. The architecture uses a shared representation trunk with six separate sigmoid output heads — one per horizon.

**Key training details:**
- Input features standardised with `StandardScaler`
- Optimiser: Adam (`lr=0.0001`)
- Loss: binary cross-entropy per output head
- Callbacks: `EarlyStopping` (patience=20), `ReduceLROnPlateau`
- Up to 200 epochs with 20% validation split

**Explainability (DL):**
- **Integrated Gradients** — measures feature attribution by integrating gradients along the path from a zero baseline to the input; produces global and per-patient importance rankings
- **SHAP** (permutation algorithm) — computed per horizon using a single-output model wrapper; global summary plots and individual waterfall plots
- **LIME** — via a custom `MultiTaskLIMEExplainer` class that explains each horizon output independently

---

## Results Summary

The table below shows AUC-ROC scores across prediction horizons:

| Horizon | ML (XGBoost) | DL (Multi-task NN) |
|---|---|---|
| Day 1 | 96.68% | 96.70% |
| Day 15 | 72.00% | 85.50% |
| Day 30 | 71.17% | 83.80% |
| Day 45 | 64.55% | 80.50% |
| Day 60 | 68.81% | 79.50% |
| Day 90 | 68.81% | 78.20% |

The deep learning model generalises better at longer horizons, while both models achieve near-equivalent performance at the 1-day horizon.

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
tensorflow / keras
sentence-transformers
shap
lime
matplotlib
seaborn
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Data Access

This project uses **MIMIC-IV**, a restricted-access clinical database. To reproduce this work you must:

1. Complete the [CITI Program](https://about.citiprogram.org/) training for human subjects research
2. Apply for access at [PhysioNet](https://physionet.org/content/mimiciv/2.2/)


---

## Citation

If you use this work, please cite the MIMIC-IV dataset:

> Johnson, A. E., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A., Horng, S., ... & Mark, R. G. (2023). MIMIC-IV, a freely accessible electronic health record dataset. Scientific data, 10(1), 1.