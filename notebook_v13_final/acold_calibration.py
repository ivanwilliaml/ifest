"""
A_cold calibration study. Not a new feature -- tests whether the DECISION LAYER on top of
the existing 71 features is leaving Macro F1 on the table:

  arm A  single model   vs  bagged (mean proba over M seeds)   -> probability quality
  policy P1 argmax threshold on the SAME OOF it is scored on   -> current production (optimistic)
  policy P2 absolute threshold averaged over OTHER seeds       -> honest, tests threshold overfit
  policy P3 quantile cutoff averaged over OTHER seeds          -> honest, robust to proba-scale shift

P1 is reported only as an optimistic reference; P2/P3 are the honest numbers to compare
against each other. Everything else (71 features, iterations/depth/lr, class_weights=[1,1],
StratifiedGroupKFold(5, content_hash), cold-subset evaluation) is unchanged.
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
print(f"features ready in {time.time()-t0:.1f}s, n_features={len(ALL_A_FEATS)}", flush=True)
groups = train['ch'].values
Xa = Ftr[ALL_A_FEATS].values
cold_mask = train['ch'].map(train['ch'].value_counts()).eq(1).values
cy = y[cold_mask]

SEEDS = [7, 42, 123, 2026, 777]
M_BAG = 3
GRID = np.arange(0.05, 0.96, 0.025)

def macro_at(p, thr):
    return f1_score(cy, (p[cold_mask] >= thr).astype(int), average='macro')

def best_thr(p):
    scores = [macro_at(p, t) for t in GRID]
    i = int(np.argmax(scores))
    return GRID[i], scores[i]

def frac_neg(p, thr):
    """fraction of cold rows labelled 0 at this threshold -- the quantile a rank rule targets"""
    return float((p[cold_mask] < thr).mean())

def macro_at_quantile(p, q):
    cp = p[cold_mask]
    if q <= 0: return f1_score(cy, np.ones_like(cy), average='macro'), 1.0
    thr = np.quantile(cp, q)
    return f1_score(cy, (cp >= thr).astype(int), average='macro'), thr

# ---- build OOF per seed, single and bagged, from the same fits -------------------------
oof_single, oof_bag = {}, {}
for seed in SEEDS:
    ts = time.time()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    s1 = np.zeros(len(y)); sb = np.zeros(len(y))
    for a, b in sgkf.split(train, y, groups):
        preds = []
        for mi in range(M_BAG):
            m = cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                                       class_weights=[1, 1], random_seed=seed * 100 + mi,
                                       verbose=False)
            m.fit(Xa[a], y[a])
            preds.append(m.predict_proba(Xa[b])[:, 1])
        s1[b] = preds[0]
        sb[b] = np.mean(preds, axis=0)
    oof_single[seed] = s1; oof_bag[seed] = sb
    t1, m1 = best_thr(s1); t2, m2 = best_thr(sb)
    print(f"seed {seed}: single argmax {m1:.4f} @{t1:.3f} | bagged argmax {m2:.4f} @{t2:.3f}"
          f"  ({time.time()-ts:.0f}s)", flush=True)

# ---- evaluate policies with leave-one-seed-out ----------------------------------------
def evaluate(oof):
    rows = {'P1_own_argmax': [], 'P2_abs_from_others': [], 'P3_quantile_from_others': []}
    f10 = {k: [] for k in rows}
    for s in SEEDS:
        p = oof[s]
        others = [o for o in SEEDS if o != s]
        t_own, m_own = best_thr(p)
        rows['P1_own_argmax'].append(m_own)
        f10['P1_own_argmax'].append(f1_score(cy, (p[cold_mask] >= t_own).astype(int), pos_label=0))

        t_others = np.mean([best_thr(oof[o])[0] for o in others])
        m_abs = macro_at(p, t_others)
        rows['P2_abs_from_others'].append(m_abs)
        f10['P2_abs_from_others'].append(f1_score(cy, (p[cold_mask] >= t_others).astype(int), pos_label=0))

        q_others = np.mean([frac_neg(oof[o], best_thr(oof[o])[0]) for o in others])
        m_q, _ = macro_at_quantile(p, q_others)
        rows['P3_quantile_from_others'].append(m_q)
        thr_q = np.quantile(p[cold_mask], q_others)
        f10['P3_quantile_from_others'].append(f1_score(cy, (p[cold_mask] >= thr_q).astype(int), pos_label=0))
    return rows, f10

print(f"\n{'arm':>8} {'policy':>26} {'macroF1 mean':>13} {'std':>7} {'F1-0 mean':>10}")
summary = {}
for arm, oof in (('single', oof_single), ('bagged', oof_bag)):
    rows, f10 = evaluate(oof)
    for k in rows:
        mm = np.mean(rows[k]); ss = np.std(rows[k], ddof=1); ff = np.mean(f10[k])
        summary[(arm, k)] = (mm, ss, ff, rows[k])
        print(f"{arm:>8} {k:>26} {mm:13.4f} {ss:7.4f} {ff:10.4f}", flush=True)

print("\nper-seed detail (honest policies only):")
for arm in ('single', 'bagged'):
    for k in ('P2_abs_from_others', 'P3_quantile_from_others'):
        vals = ' '.join(f"{v:.4f}" for v in summary[(arm, k)][3])
        print(f"  {arm:>6} {k:>26}: {vals}")

# reference values a production change would need to beat
base = summary[('single', 'P2_abs_from_others')][0]
for (arm, k), (mm, ss, ff, _) in summary.items():
    if (arm, k) == ('single', 'P2_abs_from_others'): continue
    if k == 'P1_own_argmax': continue
    print(f"delta vs single/P2  ->  {arm}/{k}: {mm-base:+.4f}")

print(f"\nmean quantile implied by argmax thresholds: "
      f"{np.mean([frac_neg(oof_single[s], best_thr(oof_single[s])[0]) for s in SEEDS]):.4f}")
print(f"mean argmax threshold (single): {np.mean([best_thr(oof_single[s])[0] for s in SEEDS]):.4f}")
print(f"TOTAL wall time: {time.time()-t0:.1f}s")
