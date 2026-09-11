"""Quick precision audit on Patch A features BEFORE the full CV run: check global firing
rate + label correlation for argument_binding_conflict and claim_slot_cooccurrence, plus
read a sample of fired rows to eyeball precision (lighter than a full 100+100 manual read,
but same spirit: don't commit to CV blind)."""
import io, os, json
import numpy as np, pandas as pd

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
stop_idx = next(i for i, s in enumerate(cells) if 'StratifiedGroupKFold' in s)
src = '\n\n'.join(cells[:stop_idx])
src = src.replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
ns = {'__name__': '__main__'}
exec(compile(src, os.path.join(d, '_dryrun_patcha.py'), 'exec'), ns)

train = ns['train']; Ftr = ns['Ftr']; y = ns['y']
cold_mask = train['ch'].map(train['ch'].value_counts()).eq(1).values
cold_idx = np.where(cold_mask)[0]
cold_train = train.iloc[cold_idx].reset_index(drop=True)
cold_y = y[cold_idx]
cold_feat = Ftr.iloc[cold_idx].reset_index(drop=True)

for feat in ['argument_binding_conflict', 'argument_role_alignment', 'predicate_subject_support',
             'predicate_object_support', 'claim_slot_cooccurrence', 'claim_slot_compactness']:
    v = cold_feat[feat].values
    print(f"{feat}: mean={v.mean():.4f} std={v.std():.4f}")

print()
for feat in ['argument_binding_conflict', 'argument_role_alignment']:
    v = cold_feat[feat].values
    fired = v > 0
    print(f"--- {feat} ---")
    print(f"  fires on {fired.sum()}/{len(v)} rows ({fired.mean()*100:.2f}%)")
    if fired.sum() > 0:
        print(f"  label=0 rate when fired: {(cold_y[fired]==0).mean():.4f}  (base rate: {(cold_y==0).mean():.4f})")

print("\n=== sample rows where argument_binding_conflict fires ===")
mask = cold_feat['argument_binding_conflict'].values > 0
idxs = np.where(mask)[0][:10]
for i in idxs:
    print(f"[{cold_train.loc[i,'id']}] true_label={cold_y[i]}")
    print(f"  TITLE: {cold_train.loc[i,'T']}")
    print(f"  BODY(0:200): {cold_train.loc[i,'C'][:200]}")
    print()
