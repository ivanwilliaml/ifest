"""
Diagnostic-only: estimate combined submission Macro F1 from honest local CV, using the
ACTUAL confusion matrices per regime (not a naive average of per-regime macro-F1 numbers,
which is not linear and would misrepresent the combination).
  - C_exact: provably perfect (0 label conflicts validated in the notebook itself)
  - B (has_pos + no_pos): honest 5-fold StratifiedKFold CV on the B_df training pool,
    using the exact same CatBoost config + B_THR the notebook uses
  - A_cold: reuse the notebook's own OOF confusion matrix, scaled to the test A_cold count
No test labels are used anywhere in this script.
"""
import io, os, json
import numpy as np, pandas as pd

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
src = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
src = src.replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
src = src.replace("/kaggle/working/submission.csv", os.path.join(d, 'local_out', 'submission.csv').replace("\\", "\\\\"))
ns = {'__name__': '__main__'}
exec(compile(src, os.path.join(d, '_dryrun_combest.py'), 'exec'), ns)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
import catboost as cb

B_df = ns['B_df']; B_feat_cols = ns['B_feat_cols']; SEED = ns['SEED']
Xb = B_df[B_feat_cols].values; yb = B_df['label'].values
B_THR = ns['B_THR']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
b_oof = np.zeros(len(yb))
for tr, te in skf.split(Xb, yb):
    m = cb.CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05,
                               class_weights=[1, 1], random_seed=SEED, verbose=False)
    m.fit(Xb[tr], yb[tr])
    b_oof[te] = m.predict_proba(Xb[te])[:, 1]
b_pred = (b_oof >= B_THR).astype(int)
b_cm = confusion_matrix(yb, b_pred, labels=[0, 1])
b_macro = f1_score(yb, b_pred, labels=[0, 1], average='macro')
print(f"B regime honest 5-fold CV (n={len(yb)}, threshold={B_THR}): Macro F1={b_macro:.4f}")
print("B confusion matrix [[TN,FP],[FN,TP]]:")
print(b_cm)

# A_cold: reuse notebook's own OOF confusion matrix (already honest, grouped CV), scale
# down to the ACTUAL test A_cold row count so it combines proportionally.
cold_y = ns['cold_y']; cold_oof = ns['cold_oof']; A_THR = ns['A_THR']
a_pred = (cold_oof >= A_THR).astype(int)
a_cm_oof = confusion_matrix(cold_y, a_pred, labels=[0, 1]).astype(float)
n_test_a = int((ns['reg_test'] == 'A_cold').sum())
a_cm_scaled = a_cm_oof * (n_test_a / a_cm_oof.sum())

# C_exact: provably perfect on the ACTUAL test C_exact rows -- distribute by the observed
# label mix among those rows (copied straight from train, 0 conflicts).
c_exact_pred = ns['c_exact_pred']
c_vals = c_exact_pred[~np.isnan(c_exact_pred)]
n_c_pos = int((c_vals == 1).sum()); n_c_neg = int((c_vals == 0).sum())
c_cm = np.array([[n_c_neg, 0], [0, n_c_pos]], dtype=float)  # perfect: TN=neg,TP=pos,FP=FN=0

# B: scale its CV confusion matrix (n=len(yb)) to the actual test B row count.
n_test_b = int(ns['b_mask_test'].sum())
b_cm_scaled = b_cm.astype(float) * (n_test_b / b_cm.sum())

combined = c_cm + b_cm_scaled + a_cm_scaled
TN, FP = combined[0]; FN, TP = combined[1]
prec0 = TN / max(TN + FN, 1e-9); rec0 = TN / max(TN + FP, 1e-9)
f1_0 = 2 * prec0 * rec0 / max(prec0 + rec0, 1e-9)
prec1 = TP / max(TP + FP, 1e-9); rec1 = TP / max(TP + FN, 1e-9)
f1_1 = 2 * prec1 * rec1 / max(prec1 + rec1, 1e-9)
macro = (f1_0 + f1_1) / 2

print("\n=== Combined regime-weighted confusion matrix (scaled to actual test-set sizes) ===")
print(f"  C_exact  n={len(c_vals):5d}  (perfect by construction)")
print(f"  B        n={n_test_b:5d}  (honest 5-fold CV, Macro F1={b_macro:.4f})")
print(f"  A_cold   n={n_test_a:5d}  (honest grouped OOF, Macro F1={ns['best'][1]:.4f})")
print(f"  total test rows covered: {len(c_vals)+n_test_b+n_test_a} / {len(ns['test'])}")
print(f"\ncombined confusion matrix: [[TN={TN:.1f}, FP={FP:.1f}], [FN={FN:.1f}, TP={TP:.1f}]]")
print(f"F1-0 = {f1_0:.4f}   F1-1 = {f1_1:.4f}")
print(f"ESTIMATED combined Macro F1 (local, honest CV, not a real submission) = {macro:.4f}")

naive = (len(c_vals)*1.0 + n_test_b*b_macro + n_test_a*ns['best'][1]) / (len(c_vals)+n_test_b+n_test_a)
print(f"\n(for reference) naive size-weighted AVERAGE of per-regime macro-F1s = {naive:.4f}")
print("(this naive number is NOT how macro-F1 combines across subpopulations -- shown only")
print(" to make the gap explicit; the confusion-matrix combination above is the correct one)")
