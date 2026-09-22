"""
Experiment: role_reversal_v1 (A_cold claim-conflict channel)

Hypothesis (see EXPERIMENTS.md write-up for full record):
  The manual FP audit (see CLAUDE.md, this project) found subject/object role reversal
  ("pasien kaget lihat dokter" vs the reverse) as an under-captured false-positive category,
  distinct from entity substitution (already covered by entity_conflict/entity_role_v2) and
  from predicate-argument non-binding (already covered by argument_binding_conflict).
  diag_role_reversal.py demonstrated the coverage/precision of a structural proxy for this
  on the CURRENT model's false positives only (a diagnostic read, not a trained feature).
  This experiment generalizes that proxy into a real per-row feature computed without label
  knowledge (title has exactly 2 known ENT_POOL entities + a matched predicate; scan the body
  for the entity pair sharing a window with a matching/synonym/antonym predicate root; flag
  whether the entity order relative to the predicate is reversed between title and body) and
  measures its isolated effect on A_cold OOF Macro F1.

Expected delta: modest positive (this is a narrow, high-specificity signal -- coverage is
  necessarily small since it requires exactly 2 named entities + a resolvable predicate in
  both title and body). Consistent with every OTHER isolated claim-conflict addition in this
  project's history (entity_role_v2 +0.0035, number_conflict_v2 +0.0029) rather than a
  breakthrough.

Validation plan: ONE change only (role_reversal_conflict column added to ALL_A_FEATS,
  everything else -- CV folds, SEED, model config, threshold search -- identical to the
  committed notebook baseline) so the delta is attributable. Same StratifiedGroupKFold
  (n_splits=5, group=content_hash) folds as baseline (same SEED -> byte-identical folds).
  Baseline is re-run in this same process (not copied from the log) so the comparison is
  apples-to-apples against this exact environment/library versions.

Parent: notebook_v13_final / ifest2026_dac_v13_final.ipynb (committed A_cold OOF macroF1
  0.6908 @ threshold 0.450, class_weights=[1,1], 69 features -- see role_reversal_log.txt).
"""
import io, os, json, re, string as _s, sys
import numpy as np, pandas as pd

d = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(d, '..', 'data'))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
src = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
src = src.replace("/kaggle/input", DATA_DIR.replace("\\", "\\\\"))
src = src.replace("/kaggle/working/submission.csv", os.path.join(d, 'local_out', 'submission.csv').replace("\\", "\\\\"))
os.makedirs(os.path.join(d, 'local_out'), exist_ok=True)
ns = {'__name__': '__main__'}
exec(compile(src, os.path.join(d, '_dryrun_rr_exp.py'), 'exec'), ns)

train = ns['train']; test = ns['test']; Ftr = ns['Ftr']; Fte = ns['Fte']
y = ns['y']; ALL_A_FEATS = list(ns['ALL_A_FEATS']); SEED = ns['SEED']
find_predicate_in = ns['find_predicate_in']; find_all_predicates_in = ns['find_all_predicates_in']
ENT_POOL = ns['ENT_POOL']; SYNONYM_SET = ns['SYNONYM_SET']; ANTONYM_SET = ns['ANTONYM_SET']
cached_stem = ns['cached_stem']
cb = ns['cb']

def title_entities_and_predicate(title):
    words = title.split()
    ents = [(i, w.strip(_s.punctuation)) for i, w in enumerate(words) if w.strip(_s.punctuation) in ENT_POOL]
    if len(ents) != 2:
        return None
    ti, troot = find_predicate_in(title)
    if ti is None:
        return None
    (i1, e1), (i2, e2) = ents
    order = 'e1_first' if i1 < i2 else 'e2_first'
    return dict(e1=e1, e2=e2, troot=troot, order=order)

def body_order_for_pair(body, e1, e2, troot):
    words = body.split()
    best = None
    for i, w in enumerate(words):
        wl = w.strip(_s.punctuation).lower()
        if not wl:
            continue
        from_stem = cached_stem(wl)
        mtype = None
        if from_stem == troot: mtype = 'exact'
        elif (troot, from_stem) in SYNONYM_SET: mtype = 'synonym'
        elif (troot, from_stem) in ANTONYM_SET: mtype = 'antonym'
        if mtype is None:
            continue
        lo, hi = max(0, i - 15), min(len(words), i + 15)
        window = words[lo:hi]
        pos_e1 = next((k for k, w2 in enumerate(window) if w2.strip(_s.punctuation) == e1), None)
        pos_e2 = next((k for k, w2 in enumerate(window) if w2.strip(_s.punctuation) == e2), None)
        if pos_e1 is None or pos_e2 is None:
            continue
        order = 'e1_first' if pos_e1 < pos_e2 else 'e2_first'
        if best is None or mtype == 'exact':
            best = dict(order=order, mtype=mtype)
            if mtype == 'exact':
                break
    return best

def role_reversal_conflict(title, body):
    """1.0 if title's 2-entity order relative to a shared predicate is reversed in the
    body's matching window; 0.0 if same order or no resolvable body match; 0.5 (neutral,
    matches this project's convention for 'no evidence either way') when the title itself
    doesn't have the 2-entity + predicate shape needed to even ask the question."""
    info = title_entities_and_predicate(title)
    if info is None:
        return 0.5
    bres = body_order_for_pair(body, info['e1'], info['e2'], info['troot'])
    if bres is None:
        return 0.0
    return 1.0 if bres['order'] != info['order'] else 0.0

with_timer_start = __import__('time').time()
train['role_reversal_conflict'] = [role_reversal_conflict(t, c) for t, c in zip(train['T'], train['C'])]
test['role_reversal_conflict'] = [role_reversal_conflict(t, c) for t, c in zip(test['T'], test['C'])]
print(f"[TIMER] role_reversal_feature: {__import__('time').time()-with_timer_start:.1f}s")
print("train coverage (non-neutral, i.e. title had 2 entities + predicate):",
      (train['role_reversal_conflict'] != 0.5).mean())
print("train positive-flag rate (reversed) among non-neutral:",
      train.loc[train['role_reversal_conflict'] != 0.5, 'role_reversal_conflict'].mean())

Ftr['role_reversal_conflict'] = train['role_reversal_conflict'].values
Fte['role_reversal_conflict'] = test['role_reversal_conflict'].values

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

def run_cv(feat_cols, class_weights):
    groups = train['ch'].values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(train, y, groups))
    X_full = Ftr[feat_cols].values
    oof = np.zeros(len(train))
    for a, bidx in folds:
        m = cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                                   class_weights=class_weights, random_seed=SEED, verbose=False)
        m.fit(X_full[a], y[a])
        oof[bidx] = m.predict_proba(X_full[bidx])[:, 1]
    cnt = train['ch'].value_counts()
    cold_mask = train['ch'].map(cnt).eq(1).values
    cold_oof = oof[cold_mask]; cold_y = y[cold_mask]
    best = (0.5, -1)
    for t in np.arange(0.05, 0.96, 0.025):
        mac = f1_score(cold_y, (cold_oof >= t).astype(int), average='macro')
        if mac > best[1]: best = (t, mac)
    thr, mac = best
    pred = (cold_oof >= thr).astype(int)
    f1_0 = f1_score(cold_y, pred, pos_label=0)
    f1_1 = f1_score(cold_y, pred, pos_label=1)
    return dict(macro=mac, thr=thr, f1_0=f1_0, f1_1=f1_1, n_feats=len(feat_cols))

print("\n=== Baseline (re-run in this process, ALL_A_FEATS, class_weights=[1,1]) ===")
base = run_cv(ALL_A_FEATS, [1, 1])
print(base)

print("\n=== + role_reversal_conflict (ONE change), class_weights=[1,1] ===")
exp_feats = ALL_A_FEATS + ['role_reversal_conflict']
exp = run_cv(exp_feats, [1, 1])
print(exp)

print(f"\nDelta Macro F1: {exp['macro'] - base['macro']:+.4f}")
print(f"Delta F1-0    : {exp['f1_0'] - base['f1_0']:+.4f}")
print(f"Delta F1-1    : {exp['f1_1'] - base['f1_1']:+.4f}")
