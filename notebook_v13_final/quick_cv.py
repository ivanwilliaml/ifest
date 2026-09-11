"""
Fast single-fold screening (not the full honest 5-fold CV) for rapid feature iteration.
Uses ONE StratifiedGroupKFold split (fold 0 of 5) instead of training 5 CatBoost models --
roughly 1/5 the runtime. Noisier than the full 5-fold estimate (higher variance, single
sample), so treat this as a quick go/no-go screen: promising results here should still get
a real 5-fold confirmation (dryrun.py) before being treated as final.
"""
import io, os, json
import numpy as np, pandas as pd

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
stop_idx = next(i for i, s in enumerate(cells) if 'StratifiedGroupKFold' in s)
src = '\n\n'.join(cells[:stop_idx])
src = src.replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
ns = {'__name__': '__main__'}
exec(compile(src, os.path.join(d, '_quick_cv.py'), 'exec'), ns)

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix
import catboost as cb

train = ns['train']; Ftr = ns['Ftr']; y = ns['y']; ALL_A_FEATS = ns['ALL_A_FEATS']; SEED = ns['SEED']
groups = train['ch'].values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
a, bidx = next(iter(sgkf.split(train, y, groups)))  # just fold 0

Xa = Ftr[ALL_A_FEATS].values
m = cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                           class_weights=[4, 1], random_seed=SEED, verbose=False)
m.fit(Xa[a], y[a])
proba = m.predict_proba(Xa[bidx])[:, 1]

cold_mask = train['ch'].map(train['ch'].value_counts()).eq(1).values
fold_cold_mask = cold_mask[bidx]
cy = y[bidx][fold_cold_mask]; cp = proba[fold_cold_mask]

best = (0.5, -1)
for t in np.arange(0.05, 0.96, 0.025):
    mac = f1_score(cy, (cp >= t).astype(int), average='macro')
    if mac > best[1]: best = (t, mac)
pred = (cp >= best[0]).astype(int)
print(f"n_features: {len(ALL_A_FEATS)}")
print(f"SINGLE-FOLD screen: Macro F1 = {best[1]:.4f} @ threshold={best[0]:.3f}  (n_cold_in_fold={fold_cold_mask.sum()})")
print(f"confusion: {confusion_matrix(cy, pred, labels=[0,1]).tolist()}")
