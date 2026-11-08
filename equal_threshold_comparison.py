"""Fair threshold comparison: every model gets its own Youden threshold from out-of-fold TRAINING predictions."""
import json, io, contextlib, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close('all')
nb = json.load(open('dengue-detection4.ipynb')); g = {'__name__': '__main__'}
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code' or i >= 62: continue
    src = ''.join(c['source']).replace('/kaggle/input/datasets/musabbir22/dengue-detection-dataset4/Dengue Fever Hematological Dataset.csv', 'data/Dengue_Fever_Hematological_Dataset.csv')
    src = '\n'.join(l for l in src.split('\n') if not l.strip().startswith(('!', '%')))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, f'cell{i}', 'exec'), g)
    plt.close('all')
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
y_tr, y_te, cv, cm = g['y_train'], g['y_test'], g['cv_strategy'], g['calculate_metrics']
def youden(model, Xtr):
    p = cross_val_predict(clone(model), Xtr, y_tr, cv=cv, method='predict_proba')[:, 1]
    f, t, th = roc_curve(y_tr, p); return th[np.argmax(t - f)]
rows, preds = {}, {}
for name, m in g['best_models'].items():
    sc = name in ('Logistic Regression', 'SVM')
    Xtr = g['X_raw_train_scaled'] if sc else g['X_raw_train']; Xte = g['X_raw_test_scaled'] if sc else g['X_raw_test']
    thr = youden(m, Xtr); p = m.predict_proba(Xte)[:, 1]; yp = (p >= thr).astype(int)
    rows[name] = (thr, cm(y_te, yp, p)); preds[name] = yp
for lab, mk, Xtr, Xte in (('A raw', 'model_A', 'X_raw_train', 'X_raw_test'), ('B dev', 'model_B', 'X_dev_train', 'X_dev_test'), ('C dchss', 'model_C', 'X_dchss_train', 'X_dchss_test')):
    m = g[mk]; thr = youden(m, g[Xtr]); p = m.predict_proba(g[Xte])[:, 1]; yp = (p >= thr).astype(int)
    rows[lab] = (thr, cm(y_te, yp, p))
thrD = g['best_thresh']; pD = g['y_prob_D']; ypD = (pD >= thrD).astype(int)
rows['D final'] = (thrD, cm(y_te, ypD, pD))
for k, (t, m) in rows.items():
    print(f"{k:22s} thr={t:.3f} acc={m['Accuracy']:.3f} rec={m['Recall']:.3f} spec={m['Specificity']:.3f} f1={m['F1']:.3f} auc={m['ROC-AUC']:.3f} prauc={m['PR-AUC']:.3f} mcc={m['MCC']:.3f}")
# McNemar vs best baseline (by AUC = RF) at its Youden threshold
y = y_te.values; a = (ypD == y); b = (preds['Random Forest'] == y)
tab = [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]
print('McNemar vs RF(youden)', tab, mcnemar(tab, exact=True).pvalue)
for name in preds:
    print(name, 'cm', confusion_matrix(y, preds[name]).ravel())
