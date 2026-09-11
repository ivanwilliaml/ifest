import io, os, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import catboost as cb

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
src = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
src = src.replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
src = src.replace("/kaggle/working/submission.csv", os.path.join(d, 'local_out', 'submission.csv').replace("\\", "\\\\"))
ns = {'__name__': '__main__'}
exec(compile(src, os.path.join(d, '_dryrun_bnopos.py'), 'exec'), ns)

B_df = ns['B_df']; B_feat_cols = ns['B_feat_cols']; SEED = ns['SEED']
Xb = B_df[B_feat_cols].values; yb = B_df['label'].values
has_pos_mask = (B_df['has_pos'] == 1.0).values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(yb))
for tr, te in skf.split(Xb, yb):
    m = cb.CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05,
                               class_weights=[1, 1], random_seed=SEED, verbose=False)
    m.fit(Xb[tr], yb[tr])
    oof[te] = m.predict_proba(Xb[te])[:, 1]

m0 = ~has_pos_mask
best = (0.3, -1)
for t in np.arange(0.10, 0.96, 0.01):
    mac = f1_score(yb[m0], (oof[m0] >= t).astype(int), average='macro')
    if mac > best[1]: best = (t, mac)
print('OPTIMAL threshold for has_pos=0 (B_no_pos-like) subset alone:', best)
pred = (oof[m0] >= best[0]).astype(int)
print('confusion:', confusion_matrix(yb[m0], pred, labels=[0, 1]).tolist())

print()
m1 = has_pos_mask
best1 = (0.3, -1)
for t in np.arange(0.10, 0.96, 0.01):
    mac = f1_score(yb[m1], (oof[m1] >= t).astype(int), average='macro')
    if mac > best1[1]: best1 = (t, mac)
print('OPTIMAL threshold for has_pos=1 subset alone:', best1)

print()
print('=== combined with per-subset optimal thresholds (vs single global 0.30) ===')
pred_split = np.zeros(len(yb), dtype=int)
pred_split[m0] = (oof[m0] >= best[0]).astype(int)
pred_split[m1] = (oof[m1] >= best1[0]).astype(int)
print('overall Macro F1 (per-subset threshold):', f1_score(yb, pred_split, average='macro'))
print('overall Macro F1 (single 0.30 threshold):', f1_score(yb, (oof >= 0.30).astype(int), average='macro'))
