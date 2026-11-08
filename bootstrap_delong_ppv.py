"""M2/M5/m2: bootstrap CIs, DeLong tests, prevalence-adjusted PPV/NPV, age subgroups (seed-42 test set)."""
import pickle, numpy as np, scipy.stats as st
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef
S = pickle.load(open('outputs/state.pkl', 'rb')); y = S['y_test'].values
P = {}
for n, m in S['best_models'].items():
    X = S['X_raw_test_scaled'] if n in ('Logistic Regression', 'SVM') else S['X_raw_test']
    P[n] = m.predict_proba(X)[:, 1]
P['Final'] = S['y_prob_D']
import os
HAS_APP = os.path.exists('outputs/app_probs.npy')  # web-app test probabilities, exported separately from the backend
if HAS_APP: P['WebApp LR'] = np.load('outputs/app_probs.npy')

def delong(y, a, b):
    pos, neg = a[y == 1], a[y == 0]; pos2, neg2 = b[y == 1], b[y == 0]
    def comps(p, n):
        V10 = np.array([(np.sum(x > n) + .5 * np.sum(x == n)) / len(n) for x in p])
        V01 = np.array([(np.sum(p > x) + .5 * np.sum(p == x)) / len(p) for x in n]); return V10, V01
    a10, a01 = comps(pos, neg); b10, b01 = comps(pos2, neg2)
    auc_a, auc_b = a10.mean(), b10.mean()
    S10 = np.cov(np.vstack([a10, b10])); S01 = np.cov(np.vstack([a01, b01]))
    Sm = S10 / len(a10) + S01 / len(a01)
    var = Sm[0, 0] + Sm[1, 1] - 2 * Sm[0, 1]; z = (auc_a - auc_b) / np.sqrt(var)
    return auc_a, auc_b, 2 * st.norm.sf(abs(z))

rng = np.random.default_rng(2026); B = 2000; idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]
idx = [i for i in idx if 0 < y[i].sum() < len(i)]
print('AUC with 95% bootstrap CI')
for n, p in P.items():
    bs = [roc_auc_score(y[i], p[i]) for i in idx]
    print(f'  {n:20s} {roc_auc_score(y, p):.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]')
print('DeLong: Final vs')
for n in P:
    if n != 'Final': print(f'  {n:20s} p={delong(y, P["Final"], P[n])[2]:.3f}')
pred = S['y_pred_D']
def ss(yy, pp):
    tn, fp, fn, tp = confusion_matrix(yy, pp, labels=[0, 1]).ravel(); return tp / (tp + fn), tn / (tn + fp)
sens, spec = ss(y, pred); bs = np.array([ss(y[i], pred[i]) for i in idx])
mcc_bs = [matthews_corrcoef(y[i], pred[i]) for i in idx]
print(f'Final sens {sens:.3f} [{np.percentile(bs[:,0],2.5):.3f},{np.percentile(bs[:,0],97.5):.3f}] spec {spec:.3f} [{np.percentile(bs[:,1],2.5):.3f},{np.percentile(bs[:,1],97.5):.3f}] MCC {matthews_corrcoef(y,pred):.3f} [{np.percentile(mcc_bs,2.5):.3f},{np.percentile(mcc_bs,97.5):.3f}]')
def ppvnpv(se, sp, pr): return se * pr / (se * pr + (1 - sp) * (1 - pr)), sp * (1 - pr) / (sp * (1 - pr) + (1 - se) * pr)
if HAS_APP:
    app_pred = (P['WebApp LR'] >= 0.6683715401060241).astype(int); ase, asp = ss(y, app_pred); abs_ = np.array([ss(y[i], app_pred[i]) for i in idx])
else:
    ase, asp, abs_ = sens, spec, bs
print('PPV/NPV (Final | WebApp)')
for pr in (0.2, 0.4, y.mean()):
    a = ppvnpv(sens, spec, pr); ab = np.array([ppvnpv(s1, s2, pr) for s1, s2 in bs])
    c = ppvnpv(ase, asp, pr); cb = np.array([ppvnpv(s1, s2, pr) for s1, s2 in abs_])
    f = lambda v, arr, k: f'{v:.3f} [{np.percentile(arr[:,k],2.5):.3f},{np.percentile(arr[:,k],97.5):.3f}]'
    print(f'  prev {pr:.3f}: PPV {f(a[0],ab,0)} NPV {f(a[1],ab,1)} | app PPV {f(c[0],cb,0)} NPV {f(c[1],cb,1)}')
age = S['df_test']['age'].values
print('age dist test: <12', (age < 12).sum(), '<18', (age < 18).sum(), 'all train <18', (S['df_train']['age'] < 18).sum(), 'min', age.min())
for lab, msk in (('adult >=18', age >= 18), ('child <18', age < 18)):
    yy, pp, pr = y[msk], pred[msk], P['Final'][msk]
    se, sp = ss(yy, pp)
    print(f'  {lab}: n={msk.sum()} pos={yy.sum()} AUC={roc_auc_score(yy, pr):.3f} sens={se:.3f} spec={sp:.3f}')
