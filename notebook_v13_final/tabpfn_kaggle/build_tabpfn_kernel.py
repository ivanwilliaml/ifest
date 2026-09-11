"""Build a standalone Kaggle kernel: reuses cells 0-7 (env..claim_representation, the
confirmed-best 71-feature A_cold representation) from the main v13 notebook, then adds a
GPU TabPFN-only 5-fold comparison cell. Diagnostic/comparison only -- not a submission
notebook (TabPFN is a pretrained model, excluded from the compliant pipeline)."""
import io, json, os

d = os.path.dirname(os.path.abspath(__file__))
main_nb_path = os.path.join(os.path.dirname(d), 'ifest2026_dac_v13_final.ipynb')
nb = json.load(io.open(main_nb_path, encoding='utf-8'))
# find the raw-cell index of the claim_representation code cell (has claim_scores + Timer),
# then keep everything up to and including it (markdown cells interspersed, so index by
# content, not by a fixed code-only count).
claim_idx = next(i for i, c in enumerate(nb['cells'])
                  if c['cell_type'] == 'code' and 'def claim_scores' in ''.join(c['source']))
cells = nb['cells'][:claim_idx + 1]
print(f"keeping {len(cells)} raw cells (claim_representation at raw index {claim_idx})")

import nbformat as nbf
new_cells = list(cells)

new_cells.append(nbf.v4.new_markdown_cell("## TabPFN comparison (GPU, diagnostic only -- not part of the compliant pipeline)"))
new_cells.append(nbf.v4.new_code_cell("""
import subprocess, sys, os
subprocess.run([sys.executable,'-m','pip','install','-q','tabpfn'], check=True)
os.environ['TABPFN_TOKEN'] = 'tabpfn_sk_6zfHJtxBhCX92Lgci9Cz9GvS8qBrQfEvME8kmlO0aSI'
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'
import torch
print("CUDA available:", torch.cuda.is_available())
"""))
new_cells.append(nbf.v4.new_code_cell("""
import numpy as np, time
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from tabpfn import TabPFNClassifier

groups = train['ch'].values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(sgkf.split(train, y, groups))
Xa = Ftr[ALL_A_FEATS].values
cold_mask = train['ch'].map(train['ch'].value_counts()).eq(1).values

device = 'cuda' if torch.cuda.is_available() else 'cpu'
oof = np.zeros(len(y))
t0 = time.time()
for i,(a,b) in enumerate(folds):
    t1 = time.time()
    m = TabPFNClassifier(random_state=SEED, device=device, ignore_pretraining_limits=True)
    m.fit(Xa[a], y[a])
    oof[b] = m.predict_proba(Xa[b])[:,1]
    print(f"fold {i+1}/5 done in {time.time()-t1:.1f}s")

cy = y[cold_mask]; cp = oof[cold_mask]
best=(0.5,-1)
for t in np.arange(0.05,0.96,0.025):
    mac=f1_score(cy,(cp>=t).astype(int),average='macro')
    if mac>best[1]: best=(t,mac)
print(f"TabPFN: Macro F1 = {best[1]:.4f} @ thr={best[0]:.3f}")
print(f"(reference: CatBoost = 0.6965)")
print(f"total time: {time.time()-t0:.1f}s")
"""))

nb['cells'] = new_cells
out_path = os.path.join(d, 'tabpfn_compare.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f)
print(f"Wrote {out_path}")
