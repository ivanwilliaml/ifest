"""
Third oracle: the 51% blind spot.

Half of cold negatives have every title entity PRESENT in the body, so
entity-missing cannot see them. Hypothesis: the tampered entity is present but
PERIPHERAL -- low body frequency, late first occurrence, absent from the lead --
while a true headline's entity is the article's most salient one.
"""
import re, string, unicodedata, numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

D = r'D:\Lomba\IFEST2026_DAC\data'
train = pd.read_csv(D + r'\train.csv')
TC = 'title' if 'title' in train.columns else train.columns[1]
CC = 'content' if 'content' in train.columns else train.columns[2]
PUNC = string.punctuation + '\u2018\u2019\u201c\u201d\u2013\u2014'
norm = lambda s: re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(s))).strip()
train['T'] = train[TC].map(norm); train['C'] = train[CC].map(norm)
train['ch'] = train['C'].str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True).map(hash)
tok = lambda s: [w.strip(PUNC) for w in s.split() if w.strip(PUNC)]

cap, low = Counter(), Counter()
for s in pd.concat([train['T'], train['C']]):
    for w in tok(s):
        (cap if w[:1].isupper() else low)[w.lower()] += 1
ENT = {w for w, c in cap.items() if c >= 5 and c / (c + low.get(w, 0)) >= 0.95}

sizes = train.groupby('ch').size()
cold = train[train['ch'].map(sizes) == 1].copy()
y = cold.label.values
base = 1 - y.mean()


def sal(df):
    rows = []
    for T, C in zip(df['T'], df['C']):
        cw = [w.lower() for w in tok(C)]
        n = max(len(cw), 1)
        pos_first = {}
        tf = Counter()
        for i, w in enumerate(cw):
            tf[w] += 1
            if w not in pos_first:
                pos_first[w] = i
        bents = {w for w in cw if w in ENT}
        maxtf = max([tf[e] for e in bents], default=1)
        tents = [w.lower() for w in tok(T) if w.lower() in ENT]
        present = [e for e in tents if e in tf]
        if not present:
            rows.append(dict(min_relfreq=np.nan, min_firstpos=np.nan,
                             any_late=np.nan, any_singleton=np.nan,
                             any_out_of_lead=np.nan, n_present=0))
            continue
        relf = [tf[e] / maxtf for e in present]
        fp = [pos_first[e] / n for e in present]
        rows.append(dict(
            min_relfreq=min(relf),
            min_firstpos=max(fp),
            any_late=float(max(fp) > 0.5),
            any_singleton=float(min(tf[e] for e in present) == 1),
            any_out_of_lead=float(max(fp) > 60.0 / n),
            n_present=len(present),
        ))
    return pd.DataFrame(rows, index=df.index)


S = sal(cold)
have = S.n_present.values > 0
print('cold rows %d | with >=1 title entity present in body: %d (%.1f%%)'
      % (len(cold), have.sum(), 100 * have.mean()))

# restrict to the blind spot: every title entity present in the body
tokset = [set(w.lower() for w in tok(c)) for c in cold['C']]
tents = [[w.lower() for w in tok(t) if w.lower() in ENT] for t in cold['T']]
blind = np.array([len(te) > 0 and all(e in cs for e in te)
                  for te, cs in zip(tents, tokset)])
yb = y[blind]
print('BLIND SPOT (all title entities present): %d rows, class0=%.4f (%.1f%% of all cold neg)'
      % (blind.sum(), 1 - yb.mean(), 100 * ((y == 0) & blind).sum() / (y == 0).sum()))

print('\n--- salience signal inside the blind spot ---')
for c in ['min_relfreq', 'min_firstpos', 'any_late', 'any_singleton', 'any_out_of_lead']:
    v = S[c].values[blind].astype(float)
    ok = ~np.isnan(v)
    if ok.sum() < 50:
        continue
    auc = roc_auc_score(1 - yb[ok], v[ok])
    ap = average_precision_score(1 - yb[ok], v[ok])
    ap2 = average_precision_score(1 - yb[ok], -v[ok])
    b = 1 - yb[ok].mean()
    print('  %-16s AUC=%.4f  AP=%.4f/%.4f  base=%.4f  lift=%.2fx'
          % (c, auc, ap, ap2, b, max(ap, ap2) / b))

print('\n--- binary flags inside the blind spot: P(label=0 | flag) ---')
for c in ['any_late', 'any_singleton', 'any_out_of_lead']:
    v = S[c].values[blind]
    m = v == 1
    if m.sum() < 20:
        continue
    p0 = 1 - yb[m].mean()
    print('  %-16s fires %5d (%.1f%%)  P(0)=%.3f  lift=%.2fx  recall0=%.3f'
          % (c, m.sum(), 100 * m.mean(), p0, p0 / (1 - yb.mean()),
             ((yb == 0) & m).sum() / max((yb == 0).sum(), 1)))
