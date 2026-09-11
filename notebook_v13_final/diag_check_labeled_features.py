"""Check actual feature values (current A3-state pipeline) for the manually-labeled 'entity'
contradiction rows, to see whether the relevant conflict features are firing (nonzero) or
genuinely zero -- distinguishes 'signal missing' from 'signal present but underweighted'."""
import io, os, json
import numpy as np, pandas as pd

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
src = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
src = src.replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
src = src.replace("/kaggle/working/submission.csv", os.path.join(d, 'local_out', 'submission.csv').replace("\\", "\\\\"))
ns = {'__name__': '__main__'}
exec(compile(src, os.path.join(d, '_dryrun_checklab.py'), 'exec'), ns)

train = ns['train']; Ftr = ns['Ftr']
labeled = pd.read_csv(r'D:\Downloads\labeling_set_labeled.csv')
hi = labeled[labeled['priority'].str.startswith('HIGH')]
ent_rows = hi[(hi['relation_type'] == 'contradict') & (hi['contradiction_subtype'] == 'entity')]

cols = ['entity_conflict', 'entity_conflict_fullbody', 'canonical_entity_conflict',
        'phrase_entity_conflict', 'person_conflict', 'person_substitution',
        'any_title_ent_missing', 'n_title_ent']
id_to_idx = {v: i for i, v in enumerate(train['id'])}
rows = []
for _, r in ent_rows.iterrows():
    idx = id_to_idx.get(r['id'])
    if idx is None:
        continue
    frow = Ftr.iloc[idx]
    rows.append(dict(id=r['id'], title=r['title'][:50], **{c: frow[c] for c in cols}))
out = pd.DataFrame(rows)
print(out.to_string(index=False))
print()
print("=== summary: how many of these 38 rows have ANY conflict signal > 0 ===")
any_signal = (out[['entity_conflict','entity_conflict_fullbody','canonical_entity_conflict',
                    'phrase_entity_conflict','person_conflict','person_substitution']] > 0).any(axis=1)
print(f"rows with at least one nonzero conflict feature: {any_signal.sum()} / {len(out)}")
print(f"rows with ZERO signal across all conflict features: {(~any_signal).sum()} / {len(out)}")
