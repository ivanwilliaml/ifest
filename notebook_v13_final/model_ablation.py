"""
Final model ablation: 5 classifiers (CatBoost, LightGBM, XGBoost, ExtraTrees, TabPFN) on the
current best 71-feature A_cold representation, honest StratifiedGroupKFold(5) CV (same
groups/seed as the main pipeline). Then two ensembles:
  - top-3 average (simple mean of OOF probabilities from the 3 best individual models)
  - all-5 stacking meta-learner (LogisticRegression on the 5 models' OOF probabilities,
    itself evaluated via a SECOND honest CV layer so the meta-learner is never scored on
    rows it was fit on -- no grid-searched weights, just a learned linear combination)

COMPLIANCE NOTE: TabPFN is a pretrained tabular foundation model (transformer pretrained on
synthetic priors, weights downloaded from HuggingFace). It is included here for comparison
only -- it cannot be part of the compliant submission notebook even if it scores well,
under the same "no pretrained models" rule that already excludes IndoBERT/BERT/XLM-R.
"""
import io, os, json, time
import numpy as np, pandas as pd
import dill

d = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
with open(os.path.join(d, '_cache.dill'), 'rb') as f:
    ns = dill.load(f)
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
CLAIM_CELL_IDX = 7
ns['Ftr'] = ns['_Ftr_base'].copy(); ns['Fte'] = ns['_Fte_base'].copy()
exec(compile(cells[CLAIM_CELL_IDX], os.path.join(d, '_fast_claim.py'), 'exec'), ns)
print(f"features ready in {time.time()-t0:.1f}s, n_features={len(ns['ALL_A_FEATS'])}")

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import catboost as cb
import lightgbm as lgb
import xgboost as xgb

train = ns['train']; Ftr = ns['Ftr']; y = ns['y']; ALL_A_FEATS = ns['ALL_A_FEATS']; SEED = ns['SEED']
groups = train['ch'].values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(sgkf.split(train, y, groups))
Xa = Ftr[ALL_A_FEATS].values
cold_mask = train['ch'].map(train['ch'].value_counts()).eq(1).values

def best_macro(oof):
    cy = y[cold_mask]; cp = oof[cold_mask]
    best = (0.5, -1)
    for t in np.arange(0.05, 0.96, 0.025):
        mac = f1_score(cy, (cp >= t).astype(int), average='macro')
        if mac > best[1]: best = (t, mac)
    return best

oofs = {}; scores = {}

t1 = time.time()
oof = np.zeros(len(y))
for a, b in folds:
    m = cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03, class_weights=[4,1],
                               random_seed=SEED, verbose=False)
    m.fit(Xa[a], y[a]); oof[b] = m.predict_proba(Xa[b])[:, 1]
oofs['CatBoost'] = oof; thr, sc = best_macro(oof); scores['CatBoost'] = sc
print(f"CatBoost   : Macro F1 = {sc:.4f} @ thr={thr:.3f}  ({time.time()-t1:.1f}s)")

t1 = time.time()
oof = np.zeros(len(y))
for a, b in folds:
    m = lgb.LGBMClassifier(n_estimators=800, max_depth=6, learning_rate=0.03,
                            class_weight={0:4,1:1}, random_state=SEED, verbose=-1)
    m.fit(Xa[a], y[a]); oof[b] = m.predict_proba(Xa[b])[:, 1]
oofs['LightGBM'] = oof; thr, sc = best_macro(oof); scores['LightGBM'] = sc
print(f"LightGBM   : Macro F1 = {sc:.4f} @ thr={thr:.3f}  ({time.time()-t1:.1f}s)")

t1 = time.time()
oof = np.zeros(len(y))
for a, b in folds:
    m = xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=0.03,
                           scale_pos_weight=4.0, random_state=SEED, eval_metric='logloss',
                           verbosity=0)
    m.fit(Xa[a], y[a]); oof[b] = m.predict_proba(Xa[b])[:, 1]
oofs['XGBoost'] = oof; thr, sc = best_macro(oof); scores['XGBoost'] = sc
print(f"XGBoost    : Macro F1 = {sc:.4f} @ thr={thr:.3f}  ({time.time()-t1:.1f}s)")

t1 = time.time()
oof = np.zeros(len(y))
for a, b in folds:
    m = ExtraTreesClassifier(n_estimators=800, max_depth=None, class_weight={0:4,1:1},
                              random_state=SEED, n_jobs=-1)
    m.fit(Xa[a], y[a]); oof[b] = m.predict_proba(Xa[b])[:, 1]
oofs['ExtraTrees'] = oof; thr, sc = best_macro(oof); scores['ExtraTrees'] = sc
print(f"ExtraTrees : Macro F1 = {sc:.4f} @ thr={thr:.3f}  ({time.time()-t1:.1f}s)")

t1 = time.time()
try:
    from tabpfn import TabPFNClassifier
    oof = np.zeros(len(y))
    for a, b in folds:
        m = TabPFNClassifier(random_state=SEED)
        m.fit(Xa[a], y[a])
        oof[b] = m.predict_proba(Xa[b])[:, 1]
    oofs['TabPFN'] = oof; thr, sc = best_macro(oof); scores['TabPFN'] = sc
    print(f"TabPFN     : Macro F1 = {sc:.4f} @ thr={thr:.3f}  ({time.time()-t1:.1f}s)  [NOT compliant, comparison only]")
except Exception as e:
    print(f"TabPFN failed: {type(e).__name__}: {e}")

print(f"\n=== Individual model ranking ===")
for name, sc in sorted(scores.items(), key=lambda kv: -kv[1]):
    print(f"  {name:12s} {sc:.4f}")

print(f"\n=== Ensemble: top-3 average ===")
top3 = [name for name, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:3]]
print(f"  using: {top3}")
oof_top3 = np.mean([oofs[n] for n in top3], axis=0)
thr, sc = best_macro(oof_top3)
print(f"  top-3 avg: Macro F1 = {sc:.4f} @ thr={thr:.3f}")

print(f"\n=== Ensemble: all-5 stacking meta-learner (LogisticRegression, honest 2nd-layer CV) ===")
model_names = list(oofs.keys())
stack_X = np.column_stack([oofs[n] for n in model_names])
meta_oof = np.zeros(len(y))
for a, b in folds:  # same groups/seed -> no leakage into meta-training
    meta = LogisticRegression(max_iter=1000)
    meta.fit(stack_X[a], y[a])
    meta_oof[b] = meta.predict_proba(stack_X[b])[:, 1]
thr, sc = best_macro(meta_oof)
print(f"  meta-learner (models: {model_names}): Macro F1 = {sc:.4f} @ thr={thr:.3f}")

print(f"\nTOTAL wall time: {time.time()-t0:.1f}s")
