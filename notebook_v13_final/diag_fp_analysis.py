"""
Diagnostic-only script (no model/pipeline changes). Runs the patched v13 notebook end to
end locally, then inspects the 319 A_cold OOF false positives (true=0, predicted=1) to look
for systematic patterns:
  1. entity substitution not caught by any entity-conflict feature
  2. contradiction / polarity mismatch not caught
  3. quantity substitution not caught
  4. predicate/event evidence exists but wasn't scored as conflicting
  5. conflicting evidence that sits outside the lexically-closest chunk (i.e. full-body vs
     single-chunk signal disagree)
"""
import io, os, json
import numpy as np, pandas as pd

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
src = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
src = src.replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
src = src.replace("/kaggle/working/submission.csv", os.path.join(d, 'local_out', 'submission.csv').replace("\\", "\\\\"))
ns = {'__name__': '__main__'}
exec(compile(src, os.path.join(d, '_dryrun_diag.py'), 'exec'), ns)

train = ns['train']; Ftr = ns['Ftr']; cold_mask = ns['cold_mask']
cold_oof = ns['cold_oof']; cold_y = ns['cold_y']; A_THR = ns['A_THR']
cold_pred = (cold_oof >= A_THR).astype(int)

cold_idx = np.where(cold_mask)[0]
cold_train = train.iloc[cold_idx].reset_index(drop=True)
cold_feat = Ftr.iloc[cold_idx].reset_index(drop=True)

fp_mask = (cold_y == 0) & (cold_pred == 1)
print(f"Total cold rows: {len(cold_y)}   FP (true=0, pred=1): {int(fp_mask.sum())}")

fp_train = cold_train[fp_mask].reset_index(drop=True)
fp_feat = cold_feat[fp_mask].reset_index(drop=True)
fp_oof = cold_oof[fp_mask]

cats = {k: np.zeros(len(fp_feat), dtype=bool) for k in [
    'entity_substitution_missed', 'entity_evidence_full_only',
    'contradiction_polarity', 'quantity_substitution',
    'predicate_antonym_present', 'no_predicate_evidence_found',
    'no_signal_at_all'
]}

for i in range(len(fp_feat)):
    r = fp_feat.iloc[i]
    ent_missing = r['any_title_ent_missing'] == 1
    ent_conf_lex = r['entity_conflict'] > 0
    ent_conf_full = r['entity_conflict_fullbody'] > 0
    phrase_conf = r['phrase_entity_conflict'] > 0
    neg_mismatch = r['neg_mismatch'] == 1
    polarity_neg = r['polarity_score'] < 0
    num_mismatch = r['num_mismatch'] == 1
    qty_low = (r['n_num_title'] > 0) and (r['qty_score'] < 0.5)
    pred_antonym = r['pred_score'] < 0
    no_pred_evidence = r['found_predicate_chunk'] == 0

    cats['entity_substitution_missed'][i] = ent_missing and not ent_conf_lex and not ent_conf_full and not phrase_conf
    cats['entity_evidence_full_only'][i] = (not ent_conf_lex) and ent_conf_full
    cats['contradiction_polarity'][i] = neg_mismatch or polarity_neg
    cats['quantity_substitution'][i] = num_mismatch or qty_low
    cats['predicate_antonym_present'][i] = pred_antonym
    cats['no_predicate_evidence_found'][i] = no_pred_evidence
    cats['no_signal_at_all'][i] = (
        not ent_conf_lex and not ent_conf_full and not phrase_conf and
        not neg_mismatch and not polarity_neg and not num_mismatch and not qty_low and
        not pred_antonym
    )

print("\n=== FP category breakdown (n=%d, categories can overlap) ===" % len(fp_feat))
for k, v in cats.items():
    print(f"  {k:32s} {int(v.sum()):4d}  ({v.mean()*100:5.1f}%)")

print("\n=== avg model confidence (OOF prob) per category ===")
for k, v in cats.items():
    if v.sum() > 0:
        print(f"  {k:32s} mean_proba={fp_oof[v].mean():.3f}")

def show(mask, name, n=4):
    idxs = np.where(mask)[0][:n]
    print(f"\n--- sample: {name} ---")
    for i in idxs:
        T = fp_train.loc[i, 'T']; C = fp_train.loc[i, 'C']
        print(f"  [proba={fp_oof[i]:.3f}] TITLE: {T}")
        print(f"           BODY(0:220): {C[:220]}")

show(cats['entity_substitution_missed'], 'entity_substitution_missed')
show(cats['entity_evidence_full_only'], 'entity_evidence_full_only')
show(cats['contradiction_polarity'], 'contradiction_polarity')
show(cats['quantity_substitution'], 'quantity_substitution')
show(cats['no_predicate_evidence_found'], 'no_predicate_evidence_found')
show(cats['no_signal_at_all'], 'no_signal_at_all')

out_path = os.path.join(d, 'local_out', 'fp_analysis.csv')
fp_train[['id','T']].assign(**{k: v for k, v in cats.items()}, oof_proba=fp_oof).to_csv(out_path, index=False)
print(f"\nSaved full FP table to {out_path}")
