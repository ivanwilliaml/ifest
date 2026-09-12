"""
Routing diagnostic. Production routes a test row to the strong B/C path only if its body's
MD5 (after nh() normalisation) exactly matches a train body; anything else falls to A_cold.
Question: how many A_cold test rows actually have a NEAR-duplicate body in train (small
edits / truncation / stray characters) that exact hashing misses? Those rows are getting
the weak A_cold model when they could get the ~0.98 B path.
Pure text + sparse linear algebra, no model training.
"""
import os, re, hashlib, unicodedata, time
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

t0 = time.time()
D = r"D:\Lomba\IFEST2026_DAC\data"
train = pd.read_csv(os.path.join(D, 'train.csv')); test = pd.read_csv(os.path.join(D, 'test.csv'))

def nh(x):
    x = unicodedata.normalize('NFKC', str(x)).lower(); return re.sub(r'\s+', ' ', x).strip()
for d in (train, test):
    d['ch'] = d['content'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
    d['th'] = d['title'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
train_ch = set(train['ch']); pair = set(zip(train['th'], train['ch']))
def regime(r):
    if (r['th'], r['ch']) in pair: return 'C_exact'
    if r['ch'] in train_ch: return 'B'
    return 'A_cold'
test['regime'] = test.apply(regime, axis=1)
print("test regime counts:", test['regime'].value_counts().to_dict())

# near-duplicate search: char 3-5gram tf-idf cosine, fit on train bodies, unique train bodies only
train_u = train.drop_duplicates('ch').reset_index(drop=True)
vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=300000)
Xtr = normalize(vec.fit_transform(train_u['content'].map(nh)))
a = test[test['regime'] == 'A_cold'].reset_index(drop=True)
Xte = normalize(vec.transform(a['content'].map(nh)))
S = (Xte @ Xtr.T).toarray() if Xtr.shape[0] * Xte.shape[0] < 60_000_000 else None
if S is None:
    best = np.zeros(len(a)); arg = np.zeros(len(a), dtype=int)
    for i in range(0, Xte.shape[0], 200):
        blk = (Xte[i:i+200] @ Xtr.T).toarray(); best[i:i+200] = blk.max(1); arg[i:i+200] = blk.argmax(1)
else:
    best = S.max(1); arg = S.argmax(1)
a['max_cos'] = best; a['nn_idx'] = arg
print(f"\nA_cold test rows: {len(a)}   (search over {Xtr.shape[0]} unique train bodies, {time.time()-t0:.0f}s)")
for thr in (0.99, 0.97, 0.95, 0.90, 0.85, 0.80):
    n = int((a['max_cos'] >= thr).sum())
    print(f"  max_cos >= {thr:.2f}: {n:5d}  ({n/len(a):.1%} of A_cold, {n/len(test):.1%} of all test)")

# sanity: within TRAIN, how often does a body have a near-dup that is NOT an exact dup?
Str = (Xtr @ Xtr.T)
Str.setdiag(0); tr_best = np.asarray(Str.max(1).todense()).ravel()
print(f"\nwithin train (unique bodies): near-dup (>=0.95, non-exact) share = {(tr_best>=0.95).mean():.1%}")

# what do the near-dups look like? show length ratio + a diff snippet for a few
print("\n--- examples of A_cold test rows with a near-duplicate train body ---")
ex = a[a['max_cos'] >= 0.95].sort_values('max_cos', ascending=False).head(6)
for _, r in ex.iterrows():
    tb = nh(train_u.loc[r['nn_idx'], 'content']); eb = nh(r['content'])
    # first divergence point
    k = next((i for i, (x, y) in enumerate(zip(tb, eb)) if x != y), min(len(tb), len(eb)))
    print(f"\n cos={r['max_cos']:.4f}  len test={len(eb)} train={len(tb)}  first diff @char {k}")
    print(f"   TEST : ...{eb[max(0,k-40):k+60]}...")
    print(f"   TRAIN: ...{tb[max(0,k-40):k+60]}...")
    # would the B path have labelled info? how many train titles share that body
    n_titles = int((train['ch'] == train_u.loc[r['nn_idx'], 'ch']).sum())
    print(f"   train titles on that body: {n_titles}   test title: {r['title'][:90]}")

a[['id', 'max_cos']].to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diag_near_dup_scores.csv'), index=False)
print(f"\nTOTAL {time.time()-t0:.0f}s")
