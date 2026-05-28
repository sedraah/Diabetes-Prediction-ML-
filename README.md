# Diabetes Prediction

## Overview

A classification project focused on thorough model evaluation rather than model complexity. Using the Pima Indians Diabetes Database, the workflow covers data cleaning with domain-aware imputation, a full Logistic Regression evaluation pipeline, manual confusion matrix metric derivation, ROC-AUC analysis, threshold tuning, and stratified k-fold cross-validation.

**Data source:** [Pima Indians Diabetes Database — Kaggle (UCI ML)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## Dataset

| Property | Detail |
|---|---|
| File | `diabetes.csv` |
| Rows | 768 |
| Columns | 9 (8 features + 1 target) |
| Target column | `Outcome` (0 = no diabetes, 1 = diabetes) |
| Duplicates | 0 |

### Features

| Column | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/ml) |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function score |
| `Age` | Age in years |
| `Outcome` | **Target** — 1 = diabetic |

---

## Workflow

### Part 1: Load, Clean & Split

**Inspection**
- Loaded dataset into `df`; shape confirmed as (768, 9).
- Printed `df.head()` and `df.describe()`.
- Initial missing value check: 0 NaNs (zeros used as placeholders in the raw file).
- Duplicate rows: 0.

**Domain-aware cleaning**

Physiologically impossible zero values in the following columns were replaced with `NaN` before imputation:

| Column | Zeros replaced (→ NaN) |
|---|---|
| `Glucose` | 5 |
| `BloodPressure` | 35 |
| `SkinThickness` | 227 |
| `Insulin` | 374 |
| `BMI` | 11 |

All missing values were imputed using **column medians**. After imputation, total missing values = 0. Cleaned data exported to `cleaned_diabetes.csv` (shape: 768 × 9).

**Feature / target split**
- `X = df.drop(columns=['Outcome'])` → shape (768, 8)
- `y = df['Outcome']` → shape (768,)

---

### Part 2: Classifier Evaluation

**Train/test split**
- 70/30 stratified split (`random_state=42`)
- Train: (537, 8) | Test: (231, 8)

**Model pipeline**
`StandardScaler` → `LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)`

---

## Results

### Confusion matrix (threshold = 0.50)

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN = 129 | FP = 21 |
| **Actual 1** | FN = 38 | TP = 43 |

### Classification report

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 (no diabetes) | 0.77 | 0.86 | 0.81 | 150 |
| 1 (diabetes) | 0.67 | 0.53 | 0.59 | 81 |
| **Accuracy** | | | **0.74** | 231 |

### ROC-AUC

**AUC = 0.8361**: well above the random baseline of 0.5, indicating strong discriminative ability.

### Threshold tuning

| Threshold | Accuracy | Precision | TPR (Recall) | TNR (Specificity) | FPR |
|---|---|---|---|---|---|
| 0.30 | 0.7316 | 0.5905 | 0.7654 | 0.7133 | 0.2867 |
| **0.50** | **0.7446** | **0.6719** | **0.5309** | **0.8600** | **0.1400** |
| 0.70 | 0.7229 | 0.7179 | 0.3457 | 0.9267 | 0.0733 |

Lowering the threshold (0.30) increases recall, catching more true diabetic cases, at the cost of more false positives. Raising it (0.70) increases specificity but misses more positive cases.

### 10-Fold Stratified Cross-Validation

| Metric | Mean | Std |
|---|---|---|
| Accuracy | 0.7668 | 0.0251 |
| ROC-AUC | 0.8367 | 0.0496 |

The low standard deviations confirm the model generalises consistently across folds.

---

## Key Findings

- **Zero-as-missing is a real data quality trap.** Columns like `Insulin` had 374 zero values (48.7%) that were biologically impossible, treating them as valid would have distorted every model trained on this data.
- **Accuracy alone is misleading for imbalanced classes.** With ~65% of cases being non-diabetic, a naive classifier could score ~65% accuracy while detecting zero diabetic patients. Recall, precision, and AUC tell a far more complete story.
- **Threshold tuning trades recall for precision.** At 0.30, the model catches 76.5% of diabetic cases; at 0.70, only 34.6%. The right threshold depends on the clinical cost of a missed diagnosis vs. a false alarm.
- **Cross-validation confirms stability.** The 10-fold CV accuracy (0.767 ± 0.025) and AUC (0.837 ± 0.050) are consistent with the hold-out test results, suggesting no significant overfitting.

