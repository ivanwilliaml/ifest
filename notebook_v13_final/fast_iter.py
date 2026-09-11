"""
Fast iteration: loads the cached namespace (from prep_cache.py, everything up through
base_features), re-runs ONLY the claim_representation cell (freshly read from the current
notebook, so any edits to claim_scores are picked up), then does a single-fold CatBoost
screen. Skips the ~75s of chunk_index/lsa/char_tfidf/base_features every run.

Run prep_cache.py again if cells 0-6 (anything before claim_representation) changed.
"""
import io, os, json, time
import numpy as np
import dill

d = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
with open(os.path.join(d, '_cache.dill'), 'rb') as f:
    ns = dill.load(f)
print(f"cache loaded in {time.time()-t0:.1f}s")

nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
CLAIM_CELL_IDX = 7
assert 'def claim_scores' in cells[CLAIM_CELL_IDX], "cell index drifted, update CLAIM_CELL_IDX"

ns['Ftr'] = ns['_Ftr_base'].copy()
ns['Fte'] = ns['_Fte_base'].copy()

t1 = time.time()
exec(compile(cells[CLAIM_CELL_IDX], os.path.join(d, '_fast_claim.py'), 'exec'), ns)
print(f"claim_representation re-run in {time.time()-t1:.1f}s")

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix
import catboost as cb

train = ns['train']; Ftr = ns['Ftr']; y = ns['y']; ALL_A_FEATS = ns['ALL_A_FEATS']; SEED = ns['SEED']
groups = train['ch'].values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
a, bidx = next(iter(sgkf.split(train, y, groups)))

t2 = time.time()
Xa = Ftr[ALL_A_FEATS].values
m = cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                           class_weights=[4, 1], random_seed=SEED, verbose=False)
m.fit(Xa[a], y[a])
proba = m.predict_proba(Xa[bidx])[:, 1]
print(f"catboost single-fold trained in {time.time()-t2:.1f}s")

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
print(f"TOTAL wall time: {time.time()-t0:.1f}s")
