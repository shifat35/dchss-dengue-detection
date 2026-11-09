"""Repeated hold-out: re-run the group's full notebook pipeline (cells 1-39: split, features, tuning,
selection, Youden threshold) with a different SPLIT seed, then evaluate every model at its own Youden threshold."""
import json, io, contextlib, sys, time, pickle
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close('all')
seed = int(sys.argv[1]); ADULT = len(sys.argv) > 2 and sys.argv[2] == 'adult'
nb = json.load(open('dengue-detection4.ipynb')); g = {'__name__': '__main__'}
t0 = time.time()
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code' or i > 39 or i in (8, 9, 11): continue
    src = ''.join(c['source']).replace('data/Dengue_Fever_Hematological_Dataset.csv', 'data/Dengue_Fever_Hematological_Dataset.csv')
    if i == 10 and ADULT:
        src += "\ndf = df[df['age'] >= 18].reset_index(drop=True)\n"
    if i == 12:
        src = src.replace('random_state=RANDOM_STATE', f'random_state={seed}'); assert str(seed) in src
    src = '\n'.join(l for l in src.split('\n') if not l.strip().startswith(('!', '%')))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, f'cell{i}', 'exec'), g)
    plt.close('all')
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
y_tr, y_te, cv, cm = g['y_train'], g['y_test'], g['cv_strategy'], g['calculate_metrics']
def youden(m, X):
    p = cross_val_predict(clone(m), X, y_tr, cv=cv, method='predict_proba')[:, 1]
    f, t, th = roc_curve(y_tr, p); return th[np.argmax(t - f)]
res = {}
def ev(name, m, Xtr, Xte, thr=None):
    thr = youden(m, Xtr) if thr is None else thr
    p = m.predict_proba(Xte)[:, 1]; res[name] = dict(cm(y_te, (p >= thr).astype(int), p), thr=thr)
for name, m in g['best_models'].items():
    sc = name in ('Logistic Regression', 'SVM')
    ev(name, m, g['X_raw_train_scaled' if sc else 'X_raw_train'], g['X_raw_test_scaled' if sc else 'X_raw_test'])
ev('A raw', g['model_A'], g['X_raw_train'], g['X_raw_test'])
ev('C dchss', g['model_C'], g['X_dchss_train'], g['X_dchss_test'])
ev('Final', g['final_dchss_xgb_model'], g['X_full_train'], g['X_full_test'], g['best_thresh'])
res['_selected'] = g['best_candidate_name']; res['_n'] = (len(g['y_train']), len(y_te)); res['_w'] = (g['w_ts'], g['w_ls'], g['w_hs'])
from scipy.stats import mannwhitneyu
res['_dchss_neg_higher'] = float(np.median(g['df_test'].loc[y_te.values == 0, 'DCHSS']) > np.median(g['df_test'].loc[y_te.values == 1, 'DCHSS']))
pickle.dump(res, open(f'outputs/rep_{seed}{"_adult" if ADULT else ""}.pkl', 'wb'))
print(seed, f'{time.time()-t0:.0f}s', res['_selected'], {k: round(v['ROC-AUC'], 3) for k, v in res.items() if not k.startswith('_')})
