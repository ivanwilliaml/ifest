"""
Run once (or whenever cells 0-6 change -- env/load/embedded-data/entity-mining/BM25-LSA/
char-TFIDF/base-features): caches the full namespace via dill AFTER base_features but
BEFORE claim_representation, so fast_iter.py can re-run ONLY the claim_representation cell
(where claim_scores lives, i.e. where most feature experiments actually happen) without
repaying the ~75s of chunk_index/lsa/char_tfidf/base_features every time.
"""
import io, os, json, pickle
import dill

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
CLAIM_CELL_IDX = 7  # "## 7. Claim representation" -- verify below before trusting this
assert 'def claim_scores' in cells[CLAIM_CELL_IDX], "cell index drifted, update CLAIM_CELL_IDX"

src = '\n\n'.join(cells[:CLAIM_CELL_IDX])
src = src.replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
ns = {'__name__': '__main__'}
exec(compile(src, os.path.join(d, '_prep_cache.py'), 'exec'), ns)

# keep pristine copies of Ftr/Fte as they are pre-claim-representation, since the claim cell
# does Ftr=pd.concat([Ftr,CRtr]) -- re-running it against a cache that already has claim
# columns would duplicate/corrupt them.
ns['_Ftr_base'] = ns['Ftr'].copy()
ns['_Fte_base'] = ns['Fte'].copy()

with open(os.path.join(d, '_cache.dill'), 'wb') as f:
    dill.dump(ns, f)
print(f"Cached {len(ns)} names. Ftr shape: {ns['Ftr'].shape}, Fte shape: {ns['Fte'].shape}")
print(f"Cache file: {os.path.join(d, '_cache.dill')}")
