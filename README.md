# D-CHSS: Interpretable Dengue Detection from Routine Complete Blood Counts

**CSE 314 — Machine Learning and Deep Learning · Group 03 · University of Asia Pacific**
Course teacher: Rashik Rahman, Assistant Professor, Department of CSE

D-CHSS (Dengue Clinical Hematological Signature Score) turns 16 routine CBC values into capped, sex-aware deviations from clinical reference ranges and groups them into three interpretable scores: thrombocytopenia (TS), leukocyte response (LS) and hemoconcentration (HS). Together with the raw values, the deviations and five hematological ratios, the signature forms a 41-feature set for an XGBoost classifier.

## Files

- `dengue-detection4.ipynb`: the research notebook (data cleaning, D-CHSS features, model training, evaluation, SHAP). Every number in our report comes from this notebook.

## Main results (303 held-out test patients)

| Model (each at its own Youden threshold) | Recall | Specificity | F1 | ROC-AUC | MCC |
|---|---|---|---|---|---|
| Random Forest (raw CBC) | 0.889 | 0.453 | 0.831 | 0.663 | 0.384 |
| XGBoost (raw CBC) | 0.909 | 0.432 | 0.838 | 0.658 | 0.396 |
| **D-CHSS + XGBoost (ours)** | **0.952** | 0.368 | **0.850** | **0.685** | **0.418** |

## Dataset

Dengue Fever Hematological Dataset (Jamalpur 250-Bedded General Hospital, 2024), Mendeley Data: https://data.mendeley.com/datasets/6fsrsk3mb8/2

## Web application

- Live app: https://dengue-two-delta.vercel.app/
- Backend: https://github.com/shifat35/dengue-backend
- Frontend: https://github.com/shifat35/dengue-frontend

## Team

| Name | ID |
|---|---|
| Tashnimul Hasan Shifat | 23101142 |
| Sultan Ruhul Kuddus Nayan | 23101159 |
| Anjuma Alam | 23101164 |
| Fairoj Salim | 23101165 |

Research and teaching project only; not a medical diagnostic tool.
