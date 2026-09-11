"""
Priority 1+2 diagnostic (BEFORE fold-safe CV engineering): in-domain phrase mining from
TRAIN's own positive/negative pairs.

Experiment A (positive alignment mining): for label=1 rows where the title HAS a known
predicate root but the existing dictionary pipeline finds NO evidence in the body
(found_predicate_chunk==0), collect all body content-word stems as candidate aligned
counterparts. A (title_root, body_root) pair that recurs across >= MIN_FREQ different
articles is kept as a mined "positive alignment" pair.

Experiment B (contradiction mining): same but for label=0 rows, additionally filtered to
rows where entity_support is high and num_mismatch==0 (so entity/number swap is unlikely to
be the actual cause of inconsistency -- isolates predicate-swap-driven negatives).

This uses the FULL train set (no fold-splitting yet) purely to inspect what gets mined and
manually audit precision, per the explicit instruction: audit before CV, and fold-safety is
only needed once we commit to using this as a real feature in the CV loop.
"""
import io, os, json, re, string as _s
from collections import Counter
import numpy as np, pandas as pd

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
src = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
src = src.replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
nb_cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
stop_idx = next(i for i, s in enumerate(nb_cells) if 'StratifiedGroupKFold' in s)
src = '\n\n'.join(nb_cells[:stop_idx]).replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
ns = {'__name__': '__main__'}
exec(compile(src, os.path.join(d, '_dryrun_ab_mine.py'), 'exec'), ns)

train = ns['train']; Ftr = ns['Ftr']
find_predicate_in = ns['find_predicate_in']; find_predicate_chunk = ns['find_predicate_chunk']
chunks_of = ns['chunks_of']; cached_stem = ns['cached_stem']
STOP = {'yang','dan','di','ke','dari','pada','akan','ini','itu','juga','dengan','untuk',
        'oleh','atau','ada','tidak','sudah','telah','saat','hari','tahun','saja','bahwa'}

def title_root(title):
    ti, troot = find_predicate_in(title)
    return troot

def existing_dict_found(title, body):
    chunks = chunks_of(body)
    _, _, broot, _ = find_predicate_chunk(title, chunks)
    return broot is not None

def body_content_roots(body):
    out = set()
    for w in body.split():
        wl = w.strip(_s.punctuation).lower()
        if not wl or wl in STOP or len(wl) < 3:
            continue
        out.add(cached_stem(wl))
    return out

print("Mining Experiment A (positive alignment) from label=1 rows...")
posA_counts = Counter()
posA_rows = 0
for i in range(len(train)):
    if train['label'].iloc[i] != 1:
        continue
    title = train['T'].iloc[i]; body = train['C'].iloc[i]
    troot = title_root(title)
    if troot is None:
        continue
    if existing_dict_found(title, body):
        continue
    posA_rows += 1
    for broot in body_content_roots(body):
        if broot == troot:
            continue
        posA_counts[(troot, broot)] += 1

print(f"  candidate rows (label=1, has title predicate, no existing evidence): {posA_rows}")
MIN_FREQ_A = 4
mined_A = {p: c for p, c in posA_counts.items() if c >= MIN_FREQ_A}
print(f"  mined pairs (freq>={MIN_FREQ_A}): {len(mined_A)}")
top_A = sorted(mined_A.items(), key=lambda x: -x[1])[:40]
print("\n  === TOP 40 mined POSITIVE-alignment pairs (title_root -> body_root : freq) ===")
for (tr_, br_), c in top_A:
    print(f"    {tr_:20s} -> {br_:20s} : {c}")

print("\n\nMining Experiment B (contradiction) from label=0 rows (entity_support high, num_mismatch=0)...")
negB_counts = Counter()
negB_rows = 0
for i in range(len(train)):
    if train['label'].iloc[i] != 0:
        continue
    ent_sup = Ftr['entity_support'].iloc[i]
    num_mm = Ftr['num_mismatch'].iloc[i]
    if ent_sup < 0.8 or num_mm == 1:
        continue
    title = train['T'].iloc[i]; body = train['C'].iloc[i]
    troot = title_root(title)
    if troot is None:
        continue
    if existing_dict_found(title, body):
        continue
    negB_rows += 1
    for broot in body_content_roots(body):
        if broot == troot:
            continue
        negB_counts[(troot, broot)] += 1

print(f"  candidate rows (label=0, entity_support>=0.8, num_mismatch=0, has title predicate, no existing evidence): {negB_rows}")
MIN_FREQ_B = 3
mined_B = {p: c for p, c in negB_counts.items() if c >= MIN_FREQ_B}
print(f"  mined pairs (freq>={MIN_FREQ_B}): {len(mined_B)}")
top_B = sorted(mined_B.items(), key=lambda x: -x[1])[:40]
print("\n  === TOP 40 mined CONTRADICTION-candidate pairs (title_root -> body_root : freq) ===")
for (tr_, br_), c in top_B:
    print(f"    {tr_:20s} -> {br_:20s} : {c}")

# cross-check: how many mined_A pairs also appear in mined_B (would indicate the mining is
# NOT actually separating support vs conflict signal -- a red flag)
overlap = set(mined_A.keys()) & set(mined_B.keys())
print(f"\n\nOverlap between mined_A and mined_B pair sets: {len(overlap)} pairs")
if overlap:
    print("  (a pair appearing in BOTH would mean the same substitution occurs in both")
    print("   consistent AND inconsistent examples -- ambiguous, should probably be dropped)")
    for p in list(overlap)[:15]:
        print("   ", p, "| A_freq=", mined_A[p], "| B_freq=", mined_B[p])
