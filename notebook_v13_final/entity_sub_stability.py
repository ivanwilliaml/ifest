"""
Stability check for the entity_substitution_score conditional correction (gate:
score>0 & p_base>=0.80, lambda=1.0), across independent CV seeds. Same 71-feature
baseline model/features, only the StratifiedGroupKFold split + CatBoost random_seed
vary per seed (both tied to the same seed value, matching the project's existing
weight-sweep stability-check convention). Reports mean/std Macro F1 and F1-0 across
seeds, plus per-seed delta vs the reference baseline 0.6995.
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

train = ns['train']; Ftr = ns['Ftr']; y = ns['y']
ALL_A_FEATS_71 = [c for c in ns['ALL_A_FEATS'] if not c.startswith('entity_substitution') and
                  c not in ('n_entity_substitutions','strongest_entity_substitution',
                            'entity_role_conflict','entity_substitution_x_lexical')]
print(f"baseline feature count (must be 71): {len(ALL_A_FEATS_71)}")
groups = train['ch'].values
Xa = Ftr[ALL_A_FEATS_71].values
ess = Ftr['entity_substitution_score'].values
cold_mask = train['ch'].map(train['ch'].value_counts()).eq(1).values

def macro_best(p, mask):
    cp = p[mask]; cy = y[mask]
    best = (0.5, -1)
    for t in np.arange(0.05, 0.96, 0.025):
        mac = f1_score(cy, (cp >= t).astype(int), average='macro')
        if mac > best[1]: best = (t, mac)
    return best

SEEDS = [7, 42, 123, 2026, 777]
REF = 0.6995
LAM = 1.0
rows = []
for seed in SEEDS:
    ts = time.time()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(sgkf.split(train, y, groups))
    oof_base = np.zeros(len(train))
    for a, bidx in folds:
        m = cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                                   class_weights=[1, 1], random_seed=seed, verbose=False)
        m.fit(Xa[a], y[a])
        oof_base[bidx] = m.predict_proba(Xa[bidx])[:, 1]

    thr0, mac0 = macro_best(oof_base, cold_mask)
    cold_pred0 = (oof_base[cold_mask] >= thr0).astype(int)
    f1_0_base = f1_score(y[cold_mask], cold_pred0, pos_label=0)
    f1_1_base = f1_score(y[cold_mask], cold_pred0, pos_label=1)

    eps = 1e-6
    logit_base = np.log(np.clip(oof_base, eps, 1 - eps) / np.clip(1 - oof_base, eps, 1 - eps))
    gate = (ess > 0.0) & (oof_base >= 0.80)
    logit_corr = logit_base.copy()
    logit_corr[gate] -= LAM * ess[gate]
    p_corr = 1 / (1 + np.exp(-logit_corr))
    thr, mac = macro_best(p_corr, cold_mask)
    cold_pred = (p_corr[cold_mask] >= thr).astype(int)
    f1_0 = f1_score(y[cold_mask], cold_pred, pos_label=0)
    f1_1 = f1_score(y[cold_mask], cold_pred, pos_label=1)

    rows.append(dict(seed=seed, mac0=mac0, f1_0_base=f1_0_base, f1_1_base=f1_1_base,
                      mac=mac, f1_0=f1_0, f1_1=f1_1, n_gated=int(gate.sum())))
    print(f"seed={seed}: base MacroF1={mac0:.4f} F1-0={f1_0_base:.4f} | "
          f"corrected MacroF1={mac:.4f} F1-0={f1_0:.4f} | delta_vs_ref={mac-REF:+.4f} "
          f"({time.time()-ts:.1f}s)", flush=True)

macs = np.array([r['mac'] for r in rows])
f1_0s = np.array([r['f1_0'] for r in rows])
bases = np.array([r['mac0'] for r in rows])
print(f"\n{'seed':>6} {'base_MacroF1':>13} {'corr_MacroF1':>13} {'delta_vs_0.6995':>16} {'corr_F1-0':>10}")
for r in rows:
    print(f"{r['seed']:6d} {r['mac0']:13.4f} {r['mac']:13.4f} {r['mac']-REF:+16.4f} {r['f1_0']:10.4f}")

print(f"\n=== SUMMARY across {len(SEEDS)} seeds ===")
print(f"corrected Macro F1: mean={macs.mean():.4f}  std={macs.std(ddof=1):.4f}")
print(f"corrected F1-0    : mean={f1_0s.mean():.4f}  std={f1_0s.std(ddof=1):.4f}")
print(f"baseline Macro F1 : mean={bases.mean():.4f}  std={bases.std(ddof=1):.4f}")
n_positive = int((macs > REF).sum())
print(f"seeds with corrected MacroF1 > {REF}: {n_positive}/{len(SEEDS)}")
print(f"TOTAL wall time: {time.time()-t0:.1f}s")
