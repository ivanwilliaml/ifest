"""TabPFN-only comparison run (reuses the cached 71-feature representation). Diagnostic
comparison ONLY -- TabPFN cannot be part of the compliant submission notebook (pretrained
tabular foundation model, weights + auth required from a third-party service)."""
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
from sklearn.metrics import f1_score

train = ns['train']; Ftr = ns['Ftr']; y = ns['y']; ALL_A_FEATS = ns['ALL_A_FEATS']; SEED = ns['SEED']
groups = train['ch'].values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(sgkf.split(train, y, groups))
Xa = Ftr[ALL_A_FEATS].values
cold_mask = train['ch'].map(train['ch'].value_counts()).eq(1).values

from tabpfn import TabPFNClassifier
oof = np.zeros(len(y))
for i, (a, b) in enumerate(folds):
    t1 = time.time()
    m = TabPFNClassifier(random_state=SEED, ignore_pretraining_limits=True)
    m.fit(Xa[a], y[a])
    oof[b] = m.predict_proba(Xa[b])[:, 1]
    print(f"  fold {i+1}/5 done ({time.time()-t1:.1f}s)")

cy = y[cold_mask]; cp = oof[cold_mask]
best = (0.5, -1)
for t in np.arange(0.05, 0.96, 0.025):
    mac = f1_score(cy, (cp >= t).astype(int), average='macro')
    if mac > best[1]: best = (t, mac)
print(f"\nTabPFN: Macro F1 = {best[1]:.4f} @ thr={best[0]:.3f}")
print(f"(for reference: CatBoost = 0.6965)")
print(f"TOTAL wall time: {time.time()-t0:.1f}s")
