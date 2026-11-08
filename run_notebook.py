"""Re-run the group's notebook (dengue-detection4.ipynb) unchanged except for the data path,
suppressing its plots, and save the fitted state for redrawing figures at print size."""
import json, pickle, sys, io, contextlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.show = lambda *a, **k: plt.close('all')
nb = json.load(open('dengue-detection4.ipynb'))
g = {'__name__': '__main__'}
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code' or i >= 62:
        continue
    src = ''.join(c['source'])
    src = src.replace('/kaggle/input/datasets/musabbir22/dengue-detection-dataset4/Dengue Fever Hematological Dataset.csv',
                      'data/Dengue_Fever_Hematological_Dataset.csv')
    lines = [l for l in src.split('\n') if not l.strip().startswith(('!', '%'))]
    src = '\n'.join(lines)
    # notebook cells that end with a bare expression just display it; harmless under exec
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            exec(compile(src, f'cell{i}', 'exec'), g)
        except Exception as e:
            print(f'ERROR in cell {i}: {e!r}', file=sys.stderr)
            raise
    plt.close('all')
    out = buf.getvalue()
    if i in (25, 30, 34, 38, 39, 46, 47, 55):
        print(f'--- cell {i}\n' + out[:700])

keep = ['y_test', 'y_prob_D', 'y_pred_D', 'best_models', 'X_raw_test', 'X_raw_test_scaled', 'X_full_test',
        'df_test', 'df_train', 'y_train', 'best_candidate_name', 'best_thresh', 'y_pred_best_baseline',
        'best_baseline_name', 'final_dchss_xgb_model', 'shap_values', 'low_thresh', 'high_thresh']
state = {k: g[k] for k in keep if k in g}
pickle.dump(state, open('outputs/state.pkl', 'wb'))
print('saved keys', sorted(state))
