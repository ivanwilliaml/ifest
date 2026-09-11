"""
Experiment 3: specialist hard-negative correction model. Base model (weight=1, confirmed
best) produces OOF probability; a small specialist model learns to correct that probability
using base_p + existing conflict features, targeting exactly "rows the base model looks very
positive on, that a conflict signal suggests might actually be negative." Full OOF stacking:
specialist is trained ONLY on the same fold's base OOF values (never in-sample).
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
print(f"features ready in {time.time()-t0:.1f}s")

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
import catboost as cb

train = ns['train']; Ftr = ns['Ftr']; y = ns['y']; ALL_A_FEATS = ns['ALL_A_FEATS']; SEED = ns['SEED']
groups = train['ch'].values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(sgkf.split(train, y, groups))
Xa = Ftr[ALL_A_FEATS].values
cold_mask = train['ch'].map(train['ch'].value_counts()).eq(1).values
cy_all = y[cold_mask]

def macro_best(oof):
    cp = oof[cold_mask]
    best = (0.5, -1)
    for t in np.arange(0.02, 0.98, 0.01):
        mac = f1_score(cy_all, (cp >= t).astype(int), average='macro')
        if mac > best[1]: best = (t, mac)
    return best

# Step 1: base model OOF (weight=1, confirmed best from Experiment 1).
t1 = time.time()
oof_base = np.zeros(len(y))
for a, b in folds:
    m = cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                               class_weights=[1, 1], random_seed=SEED, verbose=False)
    m.fit(Xa[a], y[a])
    oof_base[b] = m.predict_proba(Xa[b])[:, 1]
thr0, mac0 = macro_best(oof_base)
print(f"base model OOF: Macro F1 = {mac0:.4f} @ thr={thr0:.3f}  ({time.time()-t1:.1f}s)")

# Step 2: specialist features = base_p + a handful of existing conflict signals.
conflict_cols = ['entity_conflict', 'entity_conflict_fullbody', 'num_mismatch',
                  'qty_context_conflict', 'neg_mismatch', 'argument_binding_conflict',
                  'claim_conflict_score', 'canonical_entity_conflict']
Xs = np.column_stack([oof_base] + [Ftr[c].values for c in conflict_cols])
print(f"specialist features: base_p + {conflict_cols}")

t1 = time.time()
oof_spec = np.zeros(len(y))
for a, b in folds:
    spec = cb.CatBoostClassifier(iterations=200, depth=3, learning_rate=0.05,
                                  random_seed=SEED, verbose=False)
    spec.fit(Xs[a], y[a])
    oof_spec[b] = spec.predict_proba(Xs[b])[:, 1]
thr_s, mac_s = macro_best(oof_spec)
print(f"specialist-corrected: Macro F1 = {mac_s:.4f} @ thr={thr_s:.3f}  ({time.time()-t1:.1f}s)")

# Step 3: also try a simple LogisticRegression specialist (per the user's spec, "small
# specialist CatBoost / Logistic Regression").
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
t1 = time.time()
oof_lr = np.zeros(len(y))
for a, b in folds:
    sc = StandardScaler().fit(Xs[a])
    lr = LogisticRegression(max_iter=1000)
    lr.fit(sc.transform(Xs[a]), y[a])
    oof_lr[b] = lr.predict_proba(sc.transform(Xs[b]))[:, 1]
thr_lr, mac_lr = macro_best(oof_lr)
print(f"specialist (LogisticRegression): Macro F1 = {mac_lr:.4f} @ thr={thr_lr:.3f}  ({time.time()-t1:.1f}s)")

print(f"\n(reference: base model alone = {mac0:.4f})")
print(f"TOTAL wall time: {time.time()-t0:.1f}s")
