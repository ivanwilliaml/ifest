"""Check person_conflict's global precision on A_cold: does it actually correlate with
true label=0, or does it fire just as often on genuine label=1 rows (explaining why CatBoost
doesn't weight it heavily despite it being 'on' for most of the hard unexplained FPs)?"""
import io, os, json
import numpy as np, pandas as pd

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
src = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
src = src.replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
src = src.replace("/kaggle/working/submission.csv", os.path.join(d, 'local_out', 'submission.csv').replace("\\", "\\\\"))
ns = {'__name__': '__main__'}
exec(compile(src, os.path.join(d, '_dryrun_pc.py'), 'exec'), ns)

train = ns['train']; Ftr = ns['Ftr']; cold_mask = ns['cold_mask']
y = ns['y']
cold_idx = np.where(cold_mask)[0]
pc = Ftr['person_conflict'].values[cold_idx]
lab = y[cold_idx]

print("person_conflict distribution on A_cold (n=%d):" % len(pc))
print(pd.Series(pc).value_counts().sort_index())
print()
for thr in [0.99, 0.5, 0.01]:
    mask = pc >= thr
    print(f"person_conflict >= {thr}: n={mask.sum()}, label=0 rate={lab[mask].mean() if mask.sum() else float('nan'):.4f} "
          f"(base rate={lab.mean():.4f})")
print()
mask1 = pc >= 0.99
print(f"Among rows with person_conflict==1.0 (n={mask1.sum()}): label=0 count={int((lab[mask1]==0).sum())}, "
      f"label=1 count={int((lab[mask1]==1).sum())}, precision(label=0 | conflict=1)={(lab[mask1]==0).mean():.4f}")
mask0 = pc < 0.01
print(f"Among rows with person_conflict==0 (n={mask0.sum()}): label=0 rate={ (lab[mask0]==0).mean():.4f}")
