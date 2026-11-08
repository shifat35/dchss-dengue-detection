"""External validation (report Section IV-E, Table VII, Fig. 13).
Data: data/Dengue_Fever_Hematological_Dataset.csv (Jamalpur, Mendeley) and data/CBC_Report_Dhaka.csv
("CBC Report.csv" from https://github.com/mritunjoychk17/Dengue-Prediction-in-Bangladesh-Using-Machine-Learning).
Run from the repository root:  python analysis/external_validation.py

External validation: develop on Jamalpur (2024), test untouched on Dhaka (Riya et al., two hospitals, 2023).
Only the 10 CBC/demographic variables recorded in both cohorts are used."""
import numpy as np, pandas as pd, json, pickle, warnings, os
os.makedirs('outputs', exist_ok=True)
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

RS = 42
# ---------- development cohort (Jamalpur) ----------
j = pd.read_csv('data/Dengue_Fever_Hematological_Dataset.csv')
j = j.rename(columns={'Gender': 'gender', 'Age': 'age', 'Hemoglobin(g/dl)': 'hemoglobin', 'Neutrophils(%)': 'neutrophils',
                      'Lymphocytes(%)': 'lymphocytes', 'Monocytes(%)': 'monocytes', 'Eosinophils(%)': 'eosinophils',
                      'RBC': 'rbc', 'Total Platelet Count(/cumm)': 'platelet', 'Total WBC count(/cumm)': 'wbc', 'Result': 'result'})
j = j.drop_duplicates().reset_index(drop=True)
j['y'] = (j['result'].str.lower().str.strip() == 'positive').astype(int)
# ---------- external cohort (Dhaka) ----------
d = pd.read_csv('data/CBC_Report_Dhaka.csv')
d = d.rename(columns={'Gender': 'gender', 'Age': 'age', 'Haemoglobin': 'hemoglobin', 'Neutrophil': 'neutrophils',
                      'Lymphocyte': 'lymphocytes', 'Monocyte': 'monocytes', 'Eosinophil': 'eosinophils', 'RBC': 'rbc',
                      'Platelets': 'platelet', 'WBC': 'wbc', 'Result': 'result'})
d['platelet'] = d['platelet'] * 1000.0      # x10^3/uL -> /cumm
d['wbc'] = d['wbc'] * 1000.0
d.loc[d['rbc'] <= 0, 'rbc'] = np.nan        # one impossible value
d['y'] = (d['result'].str.lower().str.strip() == 'positive').astype(int)
FEAT = ['hemoglobin', 'rbc', 'wbc', 'neutrophils', 'lymphocytes', 'monocytes', 'eosinophils', 'platelet']
n_raw_d = len(d)
d = d.drop_duplicates(subset=['gender', 'age'] + FEAT + ['y']).reset_index(drop=True)
for df in (j, d):
    df['gender_enc'] = (df['gender'].str.lower().str.strip() == 'male').astype(int)
med = j[['age'] + FEAT].median()
for df in (j, d):
    df[['age'] + FEAT] = df[['age'] + FEAT].fillna(med)

# ---------- reduced D-CHSS (same reference ranges and cap as the main pipeline) ----------
RR = {'hemoglobin': {1: (13.0, 17.0), 0: (12.0, 15.5)}, 'rbc': {1: (4.5, 5.5), 0: (4.0, 5.0)},
      'neutrophils': (40., 70.), 'lymphocytes': (20., 40.), 'monocytes': (2., 8.), 'eosinophils': (1., 6.),
      'platelet': (150000., 450000.), 'wbc': (4000., 11000.)}
def dev(df, p):
    r = RR[p]; g = df['gender_enc'].values
    L = np.array([r[k][0] if isinstance(r, dict) else r[0] for k in g]); U = np.array([r[k][1] if isinstance(r, dict) else r[1] for k in g])
    x = df[p].values.astype(float); o = np.zeros(len(x))
    lo, hi = x < L, x > U; o[lo] = -(L[lo] - x[lo]) / L[lo]; o[hi] = (x[hi] - U[hi]) / U[hi]
    return np.clip(o, -3, 3)
TS, LS, HS = ['platelet'], ['wbc', 'neutrophils', 'lymphocytes', 'monocytes', 'eosinophils'], ['hemoglobin', 'rbc']
for df in (j, d):
    for p in FEAT: df[p + '_dev'] = dev(df, p)
    for nm, ps in (('TS', TS), ('LS', LS), ('HS', HS)): df[nm + '_raw'] = df[[p + '_dev' for p in ps]].abs().mean(axis=1)
    e = 1e-6
    df['nlr'] = df['neutrophils'] / (df['lymphocytes'] + e); df['plr'] = df['platelet'] / (df['lymphocytes'] + e)
    df['lmr'] = df['lymphocytes'] / (df['monocytes'] + e)
sc = StandardScaler().fit(j[['TS_raw', 'LS_raw', 'HS_raw']])
for df in (j, d):
    df[['TS_std', 'LS_std', 'HS_std']] = sc.transform(df[['TS_raw', 'LS_raw', 'HS_raw']])
lr_w = LogisticRegression(max_iter=1000).fit(j[['TS_std', 'LS_std', 'HS_std']], j['y'])
w = np.abs(lr_w.coef_[0]); w = w / w.sum()
for df in (j, d):
    df['DCHSS'] = w[0] * df['TS_std'] + w[1] * df['LS_std'] + w[2] * df['HS_std']

RAW = FEAT
FULL = FEAT + [p + '_dev' for p in FEAT] + ['TS_std', 'LS_std', 'HS_std', 'DCHSS', 'nlr', 'plr', 'lmr']
COMP = ['TS_std', 'LS_std', 'HS_std', 'DCHSS']
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
spw = (j['y'] == 0).sum() / (j['y'] == 1).sum()

def tune(est, grid, X, y, n=30):
    s = RandomizedSearchCV(est, grid, n_iter=n, scoring='roc_auc', cv=cv, random_state=RS, n_jobs=-1).fit(X, y)
    return s.best_estimator_, s.best_score_
xgb_grid = {'n_estimators': [100, 200, 300, 400], 'max_depth': [2, 3, 4, 5, 6], 'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0], 'min_child_weight': [1, 3, 5, 7]}
rf_grid = {'n_estimators': [200, 300, 500], 'max_depth': [3, 5, 8, None], 'min_samples_split': [2, 5, 10],
           'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2', None]}
models = {
    'LR (raw)': (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced')), RAW, None),
    'RF (raw)': (RandomForestClassifier(random_state=RS, class_weight='balanced'), RAW, rf_grid),
    'XGBoost (raw)': (XGBClassifier(random_state=RS, eval_metric='logloss', scale_pos_weight=spw), RAW, xgb_grid),
    'D-CHSS comp. (4)': (XGBClassifier(random_state=RS, eval_metric='logloss', scale_pos_weight=spw), COMP, xgb_grid),
    'D-CHSS full (reduced)': (XGBClassifier(random_state=RS, eval_metric='logloss', scale_pos_weight=spw), FULL, xgb_grid),
}

def boot_ci(y, p, f, B=2000, seed=0):
    r = np.random.default_rng(seed); v = []
    y, p = np.asarray(y), np.asarray(p)
    for _ in range(B):
        i = r.integers(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        v.append(f(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])

def mets(y, p, t):
    yh = (p >= t).astype(int); tp = ((yh == 1) & (y == 1)).sum(); tn = ((yh == 0) & (y == 0)).sum()
    fp = ((yh == 1) & (y == 0)).sum(); fn = ((yh == 0) & (y == 1)).sum()
    se, sp = tp / (tp + fn), tn / (tn + fp)
    mcc = (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) or 1)
    return dict(acc=(tp + tn) / len(y), sens=se, spec=sp, mcc=mcc, tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))

res, probs = {}, {}
yj, yd = j['y'].values, d['y'].values
for name, (est, cols, grid) in models.items():
    if grid: est, cvs = tune(est, grid, j[cols], yj)
    oof = cross_val_predict(est, j[cols], yj, cv=cv, method='predict_proba')[:, 1]
    f, t, th = roc_curve(yj, oof); thr = th[np.argmax(t - f)]
    est.fit(j[cols], yj); p = est.predict_proba(d[cols])[:, 1]; probs[name] = p
    res[name] = dict(int_auc=roc_auc_score(yj, oof), ext_auc=roc_auc_score(yd, p),
                     ext_ci=boot_ci(yd, p, roc_auc_score).tolist(), ext_prauc=average_precision_score(yd, p),
                     brier=brier_score_loss(yd, p), thr=float(thr), **mets(yd, p, thr))
# simple clinical rule and single markers (no training)
res['Platelet < 150k (rule)'] = dict(int_auc=roc_auc_score(yj, -j['platelet']), ext_auc=roc_auc_score(yd, -d['platelet']),
                                    ext_ci=boot_ci(yd, -d['platelet'].values, roc_auc_score).tolist(),
                                    **mets(yd, (d['platelet'].values < 150000).astype(float), 0.5))
res['D-CHSS score alone'] = dict(int_auc=roc_auc_score(yj, j['DCHSS']), ext_auc=roc_auc_score(yd, d['DCHSS']))
# local ceiling: 5-fold CV inside Dhaka (shows what a locally trained model can reach)
loc = cross_val_predict(RandomForestClassifier(500, min_samples_leaf=2, random_state=RS), d[RAW], yd,
                        cv=StratifiedKFold(5, shuffle=True, random_state=RS), method='predict_proba')[:, 1]
res['_local_rf_cv_auc'] = roc_auc_score(yd, loc)
# cohort description
def desc(df):
    g = lambda c, y: df.loc[df['y'] == y, c].median()
    return {c: (g(c, 1), g(c, 0)) for c in ['age'] + FEAT} | {'n': len(df), 'pos': int(df['y'].sum()), 'male': float(df['gender_enc'].mean())}
res['_desc'] = {'jamalpur': desc(j), 'dhaka': desc(d)}
res['_n_raw_dhaka'] = n_raw_d; res['_w'] = w.tolist()
res['_dchss_neg_higher'] = {'jamalpur': float(np.median(j.loc[yj == 0, 'DCHSS']) > np.median(j.loc[yj == 1, 'DCHSS'])),
                            'dhaka': float(np.median(d.loc[yd == 0, 'DCHSS']) > np.median(d.loc[yd == 1, 'DCHSS']))}
pickle.dump(dict(res=res, probs=probs, yd=yd, platelet_d=d['platelet'].values, platelet_j=j['platelet'].values, yj=yj),
            open('outputs/external_validation.pkl', 'wb'))
print(json.dumps(res, indent=1, default=lambda o: float(o) if hasattr(o, '__float__') else str(o)))

# ---------- DeLong tests and figure ----------
import scipy.stats as st
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
def delong(y, a, b):
    def comps(p, n):
        V10 = np.array([(np.sum(x > n) + .5 * np.sum(x == n)) / len(n) for x in p])
        V01 = np.array([(np.sum(p > x) + .5 * np.sum(p == x)) / len(p) for x in n]); return V10, V01
    a10, a01 = comps(a[y == 1], a[y == 0]); b10, b01 = comps(b[y == 1], b[y == 0])
    S = np.cov(np.vstack([a10, b10])) / len(a10) + np.cov(np.vstack([a01, b01])) / len(a01)
    z = (a10.mean() - b10.mean()) / np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1]); return 2 * st.norm.sf(abs(z))
P = probs; negp = -d['platelet'].values
o = dict(platelet_j=j['platelet'].values, platelet_d=d['platelet'].values, yj=yj)
for a, b in [('D-CHSS full (reduced)', 'RF (raw)'), ('D-CHSS full (reduced)', 'XGBoost (raw)'), ('D-CHSS comp. (4)', 'XGBoost (raw)')]:
    print(f'DeLong {a} vs {b}: p = {delong(yd, P[a], P[b]):.2e}')
print(f'DeLong platelet alone vs D-CHSS full (reduced): p = {delong(yd, negp, P["D-CHSS full (reduced)"]):.2e}')
NAVY, BLUE, GREEN, LBLUE, LGREEN = '#0D47A1', '#1565C0', '#1B9E77', '#90CAF9', '#A8DCC7'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9, 'axes.titleweight': 'bold',
                     'axes.labelsize': 8.5, 'xtick.labelsize': 7.8, 'ytick.labelsize': 7.8, 'legend.fontsize': 6.9,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw={'width_ratios': [1, 1.15]})
pj, yj = o['platelet_j'] / 1000, o['yj']; pdd = o['platelet_d'] / 1000
data = [pj[yj == 0], pj[yj == 1], pdd[yd == 0], pdd[yd == 1]]
pos = [0.8, 1.6, 3.0, 3.8]
v = a1.violinplot(data, positions=pos, widths=0.7, showmedians=True, showextrema=False)
for b, c in zip(v['bodies'], [LGREEN, LBLUE, LGREEN, LBLUE]): b.set_facecolor(c); b.set_edgecolor(NAVY); b.set_alpha(0.9)
v['cmedians'].set_color(NAVY); v['cmedians'].set_linewidth(1.6)
for x, dd in zip(pos, data): a1.text(x, np.median(dd) + 18, f'{np.median(dd):.0f}', ha='center', fontsize=7, fontweight='bold', color=NAVY)
a1.axhline(150, ls='--', lw=0.9, color='#E65100'); a1.text(4.35, 158, '150', color='#E65100', fontsize=7, ha='right')
a1.set_xticks([1.2, 3.4]); a1.set_xticklabels(['Jamalpur 2024\n(development)', 'Dhaka 2023\n(external)'])
a1.set_ylabel(r'Platelet count ($\times10^3/\mu$L)'); a1.set_ylim(0, 570)
a1.scatter([], [], marker='s', c=LGREEN, label='Negative'); a1.scatter([], [], marker='s', c=LBLUE, label='Positive')
a1.legend(frameon=False, loc='upper right', ncol=2, fontsize=6.9, handletextpad=0.2, columnspacing=0.8)
a1.set_title('(a) Platelets by cohort and result')
style = {'D-CHSS full (reduced)': (BLUE, 2.3, '-'), 'D-CHSS comp. (4)': (LBLUE, 1.6, '-'), 'RF (raw)': (GREEN, 1.4, '-'),
         'XGBoost (raw)': ('#9E9E9E', 1.3, '-'), 'LR (raw)': ('#BDBDBD', 1.1, ':')}
lab = {'D-CHSS full (reduced)': 'D-CHSS feature set', 'D-CHSS comp. (4)': 'D-CHSS components (4)', 'RF (raw)': 'RF, raw values',
       'XGBoost (raw)': 'XGBoost, raw values', 'LR (raw)': 'LR, raw values'}
for k, (c, lw, ls) in style.items():
    f, t, _ = roc_curve(yd, P[k]); a2.plot(f, t, color=c, lw=lw, ls=ls, label=f'{lab[k]} ({roc_auc_score(yd, P[k]):.2f})')
f, t, _ = roc_curve(yd, negp); a2.plot(f, t, color='#E65100', lw=1.4, ls='--', label=f'Platelet count alone ({roc_auc_score(yd, negp):.2f})')
a2.plot([0, 1], [0, 1], color='#CFD8DC', lw=0.8)
a2.set_xlabel('False positive rate'); a2.set_ylabel('True positive rate'); a2.set_title('(b) ROC on the external cohort')
a2.legend(frameon=False, loc='lower right', fontsize=6.4, handlelength=1.6)
plt.tight_layout(w_pad=1.2)
plt.savefig('outputs/external_validation.png', dpi=300, bbox_inches='tight')
