"""
Builds a manual-labeling CSV for the 'hierarchical relatedness -> contradiction' classifier
idea: prioritizes the two A_cold error buckets the current heuristic features CANNOT explain
at all (no_signal_at_all, entity_substitution_missed) -- these are the rows where a human
label of the true relation/contradiction type is most likely to reveal something new,
rather than randomly sampling all 319 errors (most of which are already explained by an
existing feature bucket). Also includes a modest set of TRUE NEGATIVES the model got RIGHT
(for contrast/calibration) and a few random class-1 rows (so the labeler isn't only looking
at one class). Exports full (untruncated) title+body text -- no data leaves this machine.
"""
import io, os, json
import numpy as np, pandas as pd

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
src = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
src = src.replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
src = src.replace("/kaggle/working/submission.csv", os.path.join(d, 'local_out', 'submission.csv').replace("\\", "\\\\"))
ns = {'__name__': '__main__'}
exec(compile(src, os.path.join(d, '_dryrun_export.py'), 'exec'), ns)

train = ns['train']; Ftr = ns['Ftr']; cold_mask = ns['cold_mask']
cold_oof = ns['cold_oof']; cold_y = ns['cold_y']; A_THR = ns['A_THR']
cold_pred = (cold_oof >= A_THR).astype(int)

cold_idx = np.where(cold_mask)[0]
cold_train = train.iloc[cold_idx].reset_index(drop=True)
cold_feat = Ftr.iloc[cold_idx].reset_index(drop=True)

fp_mask = (cold_y == 0) & (cold_pred == 1)   # model wrongly says "consistent"
tn_mask = (cold_y == 0) & (cold_pred == 0)   # model correctly caught the inconsistency
pos_mask = (cold_y == 1)                      # genuinely consistent (label 1)

def bucket_flags(feat_row):
    ent_missing = feat_row['any_title_ent_missing'] == 1
    ent_conf_lex = feat_row['entity_conflict'] > 0
    ent_conf_full = feat_row['entity_conflict_fullbody'] > 0
    phrase_conf = feat_row['phrase_entity_conflict'] > 0
    neg_mismatch = feat_row['neg_mismatch'] == 1
    polarity_neg = feat_row['polarity_score'] < 0
    num_mismatch = feat_row['num_mismatch'] == 1
    qty_low = (feat_row['n_num_title'] > 0) and (feat_row['qty_score'] < 0.5)
    pred_antonym = feat_row['pred_score'] < 0
    entity_substitution_missed = ent_missing and not ent_conf_lex and not ent_conf_full and not phrase_conf
    no_signal_at_all = (
        not ent_conf_lex and not ent_conf_full and not phrase_conf and
        not neg_mismatch and not polarity_neg and not num_mismatch and not qty_low and
        not pred_antonym
    )
    return entity_substitution_missed, no_signal_at_all

rows = []
fp_idx = np.where(fp_mask)[0]
for i in fp_idx:
    esm, nsa = bucket_flags(cold_feat.iloc[i])
    if esm or nsa:
        rows.append(dict(
            priority='HIGH (unexplained FP)',
            id=cold_train.loc[i, 'id'], title=cold_train.loc[i, 'T'], body=cold_train.loc[i, 'C'],
            true_label=int(cold_y[i]), model_pred=1, model_proba=round(float(cold_oof[i]), 3),
            heuristic_bucket=('entity_substitution_missed' if esm else '') + ('+no_signal_at_all' if nsa else ''),
        ))

rng = np.random.RandomState(42)
tn_idx = np.where(tn_mask)[0]
sample_tn = rng.choice(tn_idx, size=min(20, len(tn_idx)), replace=False)
for i in sample_tn:
    rows.append(dict(
        priority='LOW (contrast: correctly-caught negative)',
        id=cold_train.loc[i, 'id'], title=cold_train.loc[i, 'T'], body=cold_train.loc[i, 'C'],
        true_label=int(cold_y[i]), model_pred=0, model_proba=round(float(cold_oof[i]), 3),
        heuristic_bucket='',
    ))

pos_idx = np.where(pos_mask)[0]
sample_pos = rng.choice(pos_idx, size=min(15, len(pos_idx)), replace=False)
for i in sample_pos:
    rows.append(dict(
        priority='LOW (contrast: genuine positive)',
        id=cold_train.loc[i, 'id'], title=cold_train.loc[i, 'T'], body=cold_train.loc[i, 'C'],
        true_label=int(cold_y[i]), model_pred=int(cold_pred[i]), model_proba=round(float(cold_oof[i]), 3),
        heuristic_bucket='',
    ))

df = pd.DataFrame(rows)
# empty columns for manual labeling
df['relation_type'] = ''       # fill: support / contradict / unrelated
df['contradiction_subtype'] = ''  # fill: entity / quantity / polarity / event_predicate / other / n_a
df['notes'] = ''

out_path = os.path.join(d, 'labeling_set.csv')
df.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"Rows to label: {len(df)}  (HIGH priority unexplained FP: {(df['priority'].str.startswith('HIGH')).sum()})")
print(f"Saved to {out_path}")
