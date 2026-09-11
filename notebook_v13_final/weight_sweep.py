"""
Experiment 1: cost-sensitive class-weight sweep + joint threshold optimization, CatBoost
only, on the current best 71-feature A_cold representation. For EACH weight, run the full
honest StratifiedGroupKFold(5) CV (same groups/seed as the main pipeline) and find that
weight's own best OOF threshold -- not just reusing the weight=4 threshold.
"""
import io, os, json, time
import numpy as np
import dill

d = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
with open(os.path.join(d, '_cache.dill'), 'rb') as f:
    ns = dill.load(f)
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
ns['Ftr'] = ns['_Ftr_base'].copy(); ns['Fte'] = ns['_Fte_base'].copy()
exec(compile(cells[7], os.path.join(d, '_fast_claim.py'), 'exec'), ns)
print(f"features ready in {time.time()-t0:.1f}s, n_features={len(ns['ALL_A_FEATS'])}")

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix
import catboost as cb

train = ns['train']; Ftr = ns['Ftr']; y = ns['y']; ALL_A_FEATS = ns['ALL_A_FEATS']; SEED = ns['SEED']
groups = train['ch'].values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(sgkf.split(train, y, groups))
Xa = Ftr[ALL_A_FEATS].values
cold_mask = train['ch'].map(train['ch'].value_counts()).eq(1).values
cy_all = y[cold_mask]

WEIGHTS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
results = []
for w in WEIGHTS:
    t1 = time.time()
    oof = np.zeros(len(y))
    for a, b in folds:
        m = cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                                   class_weights=[w, 1], random_seed=SEED, verbose=False)
        m.fit(Xa[a], y[a])
        oof[b] = m.predict_proba(Xa[b])[:, 1]
    cp = oof[cold_mask]
    best = (0.5, -1, 0, 0)
    for t in np.arange(0.02, 0.98, 0.01):
        pred = (cp >= t).astype(int)
        mac = f1_score(cy_all, pred, average='macro')
        if mac > best[1]:
            f0 = f1_score(cy_all, pred, pos_label=0)
            f1 = f1_score(cy_all, pred, pos_label=1)
            best = (t, mac, f0, f1)
    thr, mac, f0, f1 = best
    results.append(dict(weight=w, threshold=thr, F1_0=f0, F1_1=f1, MacroF1=mac))
    print(f"weight={w:3d}  thr={thr:.3f}  F1_0={f0:.4f}  F1_1={f1:.4f}  MacroF1={mac:.4f}  ({time.time()-t1:.1f}s)")

print("\n=== summary (sorted by Macro F1) ===")
for r in sorted(results, key=lambda r: -r['MacroF1']):
    print(f"  w={r['weight']:3d}  thr={r['threshold']:.3f}  F1_0={r['F1_0']:.4f}  F1_1={r['F1_1']:.4f}  MacroF1={r['MacroF1']:.4f}")

print(f"\n(reference: current production weight=4 -> 0.6965 via full StratifiedGroupKFold at 0.400 threshold)")
print(f"TOTAL wall time: {time.time()-t0:.1f}s")
