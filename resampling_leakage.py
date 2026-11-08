"""M4: how much does resampling BEFORE the train/test split inflate accuracy on this cohort?
Emulates the protocol family of [9] (stacking after class balancing); not an exact replication."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
import xgboost as xgb
df = pd.read_csv('data/Dengue_Fever_Hematological_Dataset.csv')
df['Gender'] = (df['Gender'].str.lower().str.startswith('m')).astype(int)
y_all = (df['Result'].str.lower() == 'positive').astype(int).values
X_all = df.drop(columns=['Result']).values.astype(float)
def stack(seed):
    return StackingClassifier(estimators=[('lr', make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                                          ('rf', RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)),
                                          ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=seed, eval_metric='logloss', n_jobs=1))],
                              final_estimator=LogisticRegression(max_iter=2000), cv=5, n_jobs=1)
def spec(y, p): tn, fp, fn, tp = confusion_matrix(y, p).ravel(); return tn / (tn + fp)
rows = []
for dedup in (False, True):
    X0, y0 = (X_all, y_all)
    if dedup:
        d = pd.DataFrame(X_all); d['y'] = y_all; d = d.drop_duplicates(); X0, y0 = d.drop(columns='y').values, d['y'].values
    for seed in range(5):
        for proto in ('A: ROS before split', 'B: SMOTE before split', 'C: SMOTE in training only', 'D: no resampling'):
            if proto.startswith('A'):
                X, y = RandomOverSampler(random_state=seed).fit_resample(X0, y0)
            elif proto.startswith('B'):
                X, y = SMOTE(random_state=seed).fit_resample(X0, y0)
            else:
                X, y = X0, y0
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
            if proto.startswith('C'): Xtr, ytr = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
            m = stack(seed).fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]; yp = (p >= .5).astype(int)
            rows.append(dict(dedup=dedup, seed=seed, proto=proto, acc=accuracy_score(yte, yp), auc=roc_auc_score(yte, p), sens=recall_score(yte, yp), spec=spec(yte, yp)))
        print('done', dedup, seed, flush=True)
r = pd.DataFrame(rows); r.to_csv('outputs/leak.csv', index=False)
print(r.groupby(['dedup', 'proto'])[['acc', 'auc', 'sens', 'spec']].agg(['mean', 'std']).round(3).to_string())
