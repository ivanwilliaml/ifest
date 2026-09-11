"""
Check whether the CURRENT unified B classifier (trained jointly on has_pos + no_pos rows)
is actually doing well on B_no_pos specifically, or whether the aggregate 0.984 honest CV
is dominated by the much larger/easier has_pos portion while no_pos underperforms.
No code changes -- pure diagnostic using the existing B_df pool + honest 5-fold CV.
"""
import io, os, json
import numpy as np, pandas as pd

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
src = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
src = src.replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
src = src.replace("/kaggle/working/submission.csv", os.path.join(d, 'local_out', 'submission.csv').replace("\\", "\\\\"))
ns = {'__name__': '__main__'}
exec(compile(src, os.path.join(d, '_dryrun_bsplit.py'), 'exec'), ns)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import catboost as cb

B_df = ns['B_df']; B_feat_cols = ns['B_feat_cols']; SEED = ns['SEED']
Xb = B_df[B_feat_cols].values; yb = B_df['label'].values
has_pos_mask = (B_df['has_pos'] == 1.0).values  # row-level: whether THIS body has any positive title in memory

print(f"B_df pool: {len(B_df)} rows total")
print(f"  rows where has_pos=1 (body has a known positive title): {has_pos_mask.sum()}")
print(f"  rows where has_pos=0 (no known positive -- B_no_pos-like): {(~has_pos_mask).sum()}")
print(f"  label=0 rate | has_pos=1 subset: {(yb[has_pos_mask]==0).mean():.4f}")
print(f"  label=0 rate | has_pos=0 subset: {(yb[~has_pos_mask]==0).mean():.4f}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(yb))
for tr, te in skf.split(Xb, yb):
    m = cb.CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05,
                               class_weights=[1, 1], random_seed=SEED, verbose=False)
    m.fit(Xb[tr], yb[tr])
    oof[te] = m.predict_proba(Xb[te])[:, 1]

for thr in [0.30, 0.50]:
    pred = (oof >= thr).astype(int)
    print(f"\n=== threshold={thr} ===")
    print(f"OVERALL (n={len(yb)}): Macro F1 = {f1_score(yb, pred, average='macro'):.4f}")
    m1 = has_pos_mask; m0 = ~has_pos_mask
    print(f"  has_pos=1 subset (n={m1.sum()}): Macro F1 = {f1_score(yb[m1], pred[m1], average='macro'):.4f}  "
          f"(this is the B_has_pos-like part)")
    print(f"  has_pos=0 subset (n={m0.sum()}): Macro F1 = {f1_score(yb[m0], pred[m0], average='macro'):.4f}  "
          f"(this is the B_no_pos-like part)")
    print(f"  has_pos=0 confusion: {confusion_matrix(yb[m0], pred[m0], labels=[0,1]).tolist()}")
    print(f"  has_pos=0 predicted-positive rate: {pred[m0].mean():.4f}  (true positive rate: {(yb[m0]==1).mean():.4f})")
