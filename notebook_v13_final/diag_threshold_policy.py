"""
Direction #4: threshold/decision-policy exploration. Not a new feature -- checks whether
the single global threshold (0.450) picked from the full cold-OOF sweep is actually optimal,
or whether there's headroom from a different selection criterion. Uses ONLY the existing
A_cold OOF predictions (no new features, no leakage -- same oof/cold_y already computed by
the notebook). Reports the full threshold curve and a few alternate selection criteria for
comparison, so any change is grounded in the full sweep rather than a single point estimate.
"""
import io, os, json
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
src = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
src = src.replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
src = src.replace("/kaggle/working/submission.csv", os.path.join(d, 'local_out', 'submission.csv').replace("\\", "\\\\"))
ns = {'__name__': '__main__'}
exec(compile(src, os.path.join(d, '_dryrun_thr.py'), 'exec'), ns)

cold_oof = ns['cold_oof']; cold_y = ns['cold_y']

print("=== Full threshold sweep (fine-grained) ===")
best = (0.5, -1)
rows = []
for t in np.arange(0.05, 0.96, 0.005):
    pred = (cold_oof >= t).astype(int)
    mac = f1_score(cold_y, pred, average='macro')
    rows.append((t, mac))
    if mac > best[1]:
        best = (t, mac)
print(f"best single threshold: {best[0]:.3f} -> Macro F1 = {best[1]:.4f}")

# how flat is the curve around the optimum? report +-0.05 band
near = [m for t, m in rows if abs(t - best[0]) <= 0.05]
print(f"Macro F1 range within +-0.05 of optimum: [{min(near):.4f}, {max(near):.4f}]  (flatness check)")

# repeated random split stability: does the "optimal" threshold move a lot across
# bootstrap-free repeated subsamples of the cold OOF set? (group-safe would need re-deriving
# groups; here we do a simple stratified subsample repeat as a cheap stability probe)
rng = np.random.RandomState(0)
n = len(cold_y)
opt_thrs = []
for rep in range(30):
    idx = rng.choice(n, size=n, replace=True)  # bootstrap resample of the OOF SCORES themselves
    y_b = cold_y[idx]; oof_b = cold_oof[idx]
    b_best = (0.5, -1)
    for t in np.arange(0.10, 0.91, 0.01):
        mac = f1_score(y_b, (oof_b >= t).astype(int), average='macro')
        if mac > b_best[1]:
            b_best = (t, mac)
    opt_thrs.append(b_best[0])
opt_thrs = np.array(opt_thrs)
print(f"\nbootstrap stability of optimal threshold (30 reps): mean={opt_thrs.mean():.3f}, std={opt_thrs.std():.3f}, "
      f"range=[{opt_thrs.min():.3f}, {opt_thrs.max():.3f}]")
print("(if std is large relative to the +-0.05 flatness band above, the 'optimal' threshold")
print(" is noise-dominated and not worth refining further)")
