"""
A_cold training-population study. The A_cold model is trained on ALL train rows but is only
ever deployed on cold rows (body never seen). Those two populations have very different
class balance -- cold 5.34% class-0, non-cold 18.0% class-0 (3.37x) -- so the pooled prior
is far too pessimistic for the deployment population.

Arms (folds, features, hyperparameters, evaluation all identical to production):
  T0 all rows                       -- current production
  T1 cold rows only
  T2 all rows, non-cold weighted 0.50
  T3 all rows, non-cold weighted 0.25

Evaluation is unchanged: OOF on the held-out fold's COLD rows, threshold by argmax on the
same OOF for every arm (so the comparison between arms is like-for-like), repeated over
independent seeds and reported as mean +/- std.
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

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
import catboost as cb

train = ns['train']; Ftr = ns['Ftr']; y = ns['y']; ALL_A_FEATS = ns['ALL_A_FEATS']
groups = train['ch'].values
Xa = Ftr[ALL_A_FEATS].values
cold_mask = train['ch'].map(train['ch'].value_counts()).eq(1).values
cy = y[cold_mask]
print(f"features ready in {time.time()-t0:.1f}s  n_features={len(ALL_A_FEATS)}  "
      f"cold={cold_mask.sum()} noncold={(~cold_mask).sum()}", flush=True)

GRID = np.arange(0.05, 0.96, 0.025)
SEEDS = [42, 7, 123]

def score(p):
    cp = p[cold_mask]
    best = (0.5, -1.0)
    for t in GRID:
        m = f1_score(cy, (cp >= t).astype(int), average='macro')
        if m > best[1]: best = (t, m)
    thr = best[0]
    pred = (cp >= thr).astype(int)
    return best[1], f1_score(cy, pred, pos_label=0), f1_score(cy, pred, pos_label=1), thr

ARMS = {'T0_all': None, 'T1_cold_only': 'subset', 'T2_noncold_w0.50': 0.50, 'T3_noncold_w0.25': 0.25}
results = {k: [] for k in ARMS}

for seed in SEEDS:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(sgkf.split(train, y, groups))
    for arm, cfg in ARMS.items():
        ts = time.time()
        oof = np.zeros(len(y))
        for a, b in folds:
            if cfg == 'subset':
                a_use = a[cold_mask[a]]
                w = None
            else:
                a_use = a
                w = None if cfg is None else np.where(cold_mask[a_use], 1.0, cfg)
            m = cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                                       class_weights=[1, 1], random_seed=seed, verbose=False)
            m.fit(Xa[a_use], y[a_use], sample_weight=w)
            oof[b] = m.predict_proba(Xa[b])[:, 1]
        mac, f0, f1v, thr = score(oof)
        results[arm].append((mac, f0, f1v, thr))
        print(f"seed {seed} {arm:>17}: macroF1={mac:.4f} F1-0={f0:.4f} F1-1={f1v:.4f} "
              f"thr={thr:.3f} ({time.time()-ts:.0f}s)", flush=True)

print(f"\n{'arm':>17} {'macroF1 mean':>13} {'std':>7} {'F1-0 mean':>10} {'thr mean':>9} {'delta vs T0':>12}")
base = np.mean([r[0] for r in results['T0_all']])
for arm in ARMS:
    ms = [r[0] for r in results[arm]]; fs = [r[1] for r in results[arm]]; ts_ = [r[3] for r in results[arm]]
    print(f"{arm:>17} {np.mean(ms):13.4f} {np.std(ms, ddof=1):7.4f} {np.mean(fs):10.4f} "
          f"{np.mean(ts_):9.3f} {np.mean(ms)-base:+12.4f}")
print(f"\nTOTAL wall time: {time.time()-t0:.1f}s")
