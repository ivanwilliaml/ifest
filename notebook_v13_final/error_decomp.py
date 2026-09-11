"""
Error decomposition: extract high-confidence false negatives for class 0 (y_true=0,
OOF p(class1) high) from the A_cold cold-subset OOF, using the CURRENT production
config in build_notebook.py (whatever that is when this script runs -- rebuild the
cache first with prep_cache.py if base_features changed). Dumps candidates with their
text + existing feature values to a CSV for manual categorization into the 10 error
families the user specified (entity substitution, role reversal, polarity/negation,
quantity, predicate mismatch, causality, temporal/modality, framing/exaggeration,
unsupported claim, ambiguous/label noise).
"""
import io, os, json, time
import numpy as np
import pandas as pd
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
print(f"n_features: {len(ALL_A_FEATS)}")
groups = train['ch'].values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(sgkf.split(train, y, groups))
Xa = Ftr[ALL_A_FEATS].values

oof = np.zeros(len(train))
for a, bidx in folds:
    m = cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                               class_weights=[1, 1], random_seed=SEED, verbose=False)
    m.fit(Xa[a], y[a])
    oof[bidx] = m.predict_proba(Xa[bidx])[:, 1]

cold_mask = train['ch'].map(train['ch'].value_counts()).eq(1).values
cy = y[cold_mask]; cp = oof[cold_mask]
best = (0.5, -1)
for t in np.arange(0.05, 0.96, 0.025):
    mac = f1_score(cy, (cp >= t).astype(int), average='macro')
    if mac > best[1]: best = (t, mac)
print(f"OOF Macro F1 = {best[1]:.4f} @ thr={best[0]:.3f}")

fn_mask = cold_mask & (y == 0) & (oof >= 0.8)
n80 = fn_mask.sum()
n90 = (cold_mask & (y == 0) & (oof >= 0.9)).sum()
print(f"cold class-0 rows: {cold_mask.sum() and (y[cold_mask]==0).sum()}")
print(f"high-confidence false negatives: p>=0.8 -> {n80}   p>=0.9 -> {n90}")

FEATS_OF_INTEREST = [c for c in [
    'entity_support','entity_conflict','entity_conflict_fullbody',
    'phrase_entity_support','phrase_entity_conflict',
    'person_support','person_conflict','person_substitution','person_type_match',
    'canonical_entity_support','canonical_entity_conflict',
    'qty_context_token_overlap','qty_context_exact_match','qty_context_conflict',
    'argument_binding_conflict','pred_score','polarity_score','qty_score',
    'claim_support_score','claim_conflict_score','claim_margin','found_predicate_chunk',
    'num_mismatch','neg_mismatch','bigram_ov','important_bigram_overlap','important_trigram_overlap',
    'phrase_tfidf_max','phrase_tfidf_mean','title_cov','content_cov','idf_title_cov',
] if c in Ftr.columns]

out = pd.DataFrame({'idx': np.where(fn_mask)[0], 'title': train.loc[fn_mask,'T'].values,
                     'body': train.loc[fn_mask,'C'].values, 'oof_p': oof[fn_mask]})
for c in FEATS_OF_INTEREST:
    out[c] = Ftr.loc[fn_mask, c].values
out = out.sort_values('oof_p', ascending=False).reset_index(drop=True)
out_path = os.path.join(d, 'error_decomp_fn_candidates.csv')
out.to_csv(out_path, index=False)
print(f"saved {len(out)} candidates to {out_path}")
print(f"TOTAL wall time: {time.time()-t0:.1f}s")
