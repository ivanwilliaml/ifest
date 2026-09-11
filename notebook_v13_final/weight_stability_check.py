"""Stability check: is weight=1's apparent win over weight=4 (+0.0023 in the main sweep)
real, or noise? Re-run both weights across 3 DIFFERENT StratifiedGroupKFold seeds (not the
main pipeline's SEED=42) and compare. If weight=1 doesn't consistently beat weight=4 across
seeds, treat the original result as noise-floor, per the project's established discipline."""
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

train = ns['train']; Ftr = ns['Ftr']; y = ns['y']; ALL_A_FEATS = ns['ALL_A_FEATS']
groups = train['ch'].values
Xa = Ftr[ALL_A_FEATS].values
cold_mask = train['ch'].map(train['ch'].value_counts()).eq(1).values
cy_all = y[cold_mask]

def run(weight, split_seed, model_seed):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
    oof = np.zeros(len(y))
    for a, b in sgkf.split(train, y, groups):
        m = cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                                   class_weights=[weight, 1], random_seed=model_seed, verbose=False)
        m.fit(Xa[a], y[a])
        oof[b] = m.predict_proba(Xa[b])[:, 1]
    cp = oof[cold_mask]
    best = (0.5, -1)
    for t in np.arange(0.02, 0.98, 0.01):
        mac = f1_score(cy_all, (cp >= t).astype(int), average='macro')
        if mac > best[1]: best = (t, mac)
    return best

for split_seed in [7, 123, 2026]:
    t1 = time.time()
    thr1, m1 = run(1, split_seed, split_seed)
    thr4, m4 = run(4, split_seed, split_seed)
    thr5, m5 = run(5, split_seed, split_seed)
    winner = max([('w=1', m1), ('w=4', m4), ('w=5', m5)], key=lambda kv: kv[1])
    print(f"seed={split_seed}: w=1 -> {m1:.4f} (thr={thr1:.2f}) | w=4 -> {m4:.4f} (thr={thr4:.2f}) | "
          f"w=5 -> {m5:.4f} (thr={thr5:.2f})  | best: {winner[0]}  ({time.time()-t1:.1f}s)")

print(f"\nTOTAL wall time: {time.time()-t0:.1f}s")
