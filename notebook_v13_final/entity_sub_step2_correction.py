"""
Step 2 (per user spec): conditional logit correction using entity_substitution_score,
applied ONLY when entity_substitution_score >= gate_thr AND p_base >= 0.80. Baseline
model/folds/features are UNCHANGED (71 features, class_weights=[1,1]) -- the correction
is a post-hoc adjustment of OOF probabilities, tuned only on OOF (no test leakage).
Rejects unless it improves Macro F1 AND specifically improves F1-0 vs baseline 0.6995.
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

train = ns['train']; Ftr = ns['Ftr']; y = ns['y']; SEED = ns['SEED']
ALL_A_FEATS_71 = [c for c in ns['ALL_A_FEATS'] if not c.startswith('entity_substitution') and
                  c not in ('n_entity_substitutions','strongest_entity_substitution',
                            'entity_role_conflict','entity_substitution_x_lexical')]
print(f"baseline feature count (must be 71): {len(ALL_A_FEATS_71)}")
groups = train['ch'].values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(sgkf.split(train, y, groups))
Xa = Ftr[ALL_A_FEATS_71].values

oof_base = np.zeros(len(train))
for a, bidx in folds:
    m = cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                               class_weights=[1, 1], random_seed=SEED, verbose=False)
    m.fit(Xa[a], y[a])
    oof_base[bidx] = m.predict_proba(Xa[bidx])[:, 1]

cold_mask = train['ch'].map(train['ch'].value_counts()).eq(1).values
ess = Ftr['entity_substitution_score'].values

def macro_best(p, mask):
    cp = p[mask]; cy = y[mask]
    best = (0.5, -1)
    for t in np.arange(0.05, 0.96, 0.025):
        mac = f1_score(cy, (cp >= t).astype(int), average='macro')
        if mac > best[1]: best = (t, mac)
    return best

thr0, mac0 = macro_best(oof_base, cold_mask)
cold_pred0 = (oof_base[cold_mask] >= thr0).astype(int)
f1_0_base = f1_score(y[cold_mask], cold_pred0, pos_label=0)
f1_1_base = f1_score(y[cold_mask], cold_pred0, pos_label=1)
print(f"BASELINE (71 feats, unchanged): Macro F1={mac0:.4f} @ thr={thr0:.3f}  F1-0={f1_0_base:.4f}  F1-1={f1_1_base:.4f}")

eps = 1e-6
logit_base = np.log(np.clip(oof_base, eps, 1 - eps) / np.clip(1 - oof_base, eps, 1 - eps))

results = []
for gate_thr in (0.0, 0.34, 0.5, 1.0):
    gate = (ess > gate_thr) & (oof_base >= 0.80)
    n_gated = int(gate.sum())
    for lam in (0.25, 0.5, 1.0, 2.0, 4.0):
        logit_corr = logit_base.copy()
        logit_corr[gate] -= lam * ess[gate]
        p_corr = 1 / (1 + np.exp(-logit_corr))
        thr, mac = macro_best(p_corr, cold_mask)
        cold_pred = (p_corr[cold_mask] >= thr).astype(int)
        f1_0 = f1_score(y[cold_mask], cold_pred, pos_label=0)
        f1_1 = f1_score(y[cold_mask], cold_pred, pos_label=1)
        # FN->TN recovery: rows that were false negative for class 0 (y=0, base pred=1 @thr0)
        # and are now correctly predicted 0 under the corrected threshold.
        base_pred0_mask = cold_mask & (y == 0) & (oof_base >= thr0)
        now_pred0 = (p_corr >= thr).astype(int) == 0
        recovered = int((base_pred0_mask & now_pred0).sum())
        n_base_fn = int(base_pred0_mask.sum())
        # FP increase: rows y=1 correctly predicted 1 at baseline, now flipped to 0.
        base_correct1_mask = cold_mask & (y == 1) & (oof_base >= thr0)
        now_pred1_wrong = (p_corr >= thr).astype(int) == 0
        new_fp = int((base_correct1_mask & now_pred1_wrong).sum())
        results.append(dict(gate_thr=gate_thr, lam=lam, n_gated=n_gated, mac=mac, f1_0=f1_0, f1_1=f1_1,
                             thr=thr, recovered=recovered, n_base_fn=n_base_fn, new_fp=new_fp))

results.sort(key=lambda r: -r['mac'])
print(f"\n{'gate_thr':>8} {'lam':>5} {'n_gated':>8} {'MacroF1':>8} {'F1-0':>7} {'F1-1':>7} {'thr':>6} {'FN->TN':>7} {'newFP':>6}")
for r in results[:15]:
    print(f"{r['gate_thr']:8.2f} {r['lam']:5.2f} {r['n_gated']:8d} {r['mac']:8.4f} {r['f1_0']:7.4f} {r['f1_1']:7.4f} {r['thr']:6.3f} {r['recovered']:7d} {r['new_fp']:6d}")

best = results[0]
print(f"\nBEST: gate_thr={best['gate_thr']} lam={best['lam']}  MacroF1={best['mac']:.4f} (baseline {mac0:.4f}, delta {best['mac']-mac0:+.4f})")
print(f"F1-0: {best['f1_0']:.4f} (baseline {f1_0_base:.4f}, delta {best['f1_0']-f1_0_base:+.4f})")
verdict = "GO" if (best['mac'] > mac0 and best['f1_0'] > f1_0_base) else "NO-GO (reject per spec: must improve Macro F1 AND F1-0)"
print(f"VERDICT: {verdict}")
print(f"TOTAL wall time: {time.time()-t0:.1f}s")
