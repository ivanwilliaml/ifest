"""
Follow-up oracle: the mutation census says 92% of negatives are ENTITY swaps.
So the only question that matters is: given a title entity absent from the body,
can we tell a TAMPERED entity from a merely-unmentioned one?

Hypothesis (slot competitor): a tampered entity leaves a scar -- the body still
contains the ORIGINAL entity, which is type-compatible with the swapped-in one
(same preceding word / same mined type class). An innocent missing entity has no
such competitor.
"""
import re, string, unicodedata, numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.metrics import f1_score

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

# ---- type classes: entities sharing a preceding title word are swap-compatible
prev2ent = defaultdict(set); ent2prev = defaultdict(set)
for T in train['T']:
    ws = tok(T)
    for i, w in enumerate(ws):
        if w.lower() in ENT and i > 0:
            p = ws[i - 1].lower()
            prev2ent[p].add(w.lower()); ent2prev[w.lower()].add(p)

# type class = entities reachable through a shared preceding word (precomputed)
COMPAT = defaultdict(set)
for p, es in prev2ent.items():
    if len(es) > 200:          # a generic word like "di" links everything: useless
        continue
    for e in es:
        COMPAT[e] |= es
for e in COMPAT:
    COMPAT[e].discard(e)
print('type-class map built for %d entities, median class size %d'
      % (len(COMPAT), int(np.median([len(v) for v in COMPAT.values()]))))

# ---- ground truth: which entities actually get swapped IN (from minimal pairs)
swapped_in = Counter(); swapped_out = Counter()
for h, g in train.groupby('ch'):
    pos = g[g.label == 1]['T'].tolist(); neg = g[g.label == 0]['T'].tolist()
    for p in pos:
        for n in neg:
            pt, nt = p.split(), n.split()
            op = [w for w in pt if w not in nt]; on = [w for w in nt if w not in pt]
            if len(op) == 1 and len(on) == 1:
                swapped_out[op[0].strip(PUNC).lower()] += 1
                swapped_in[on[0].strip(PUNC).lower()] += 1
print('top swapped-IN (tamper targets):', swapped_in.most_common(12))
print('distinct tamper targets:', len(swapped_in),
      ' covering %d of %d swaps' % (sum(swapped_in.values()), sum(swapped_in.values())))

sizes = train.groupby('ch').size()
cold = train[train['ch'].map(sizes) == 1].copy()
y = cold.label.values
base = 1 - y.mean()
print('\ncold rows %d, class0 %.4f' % (len(cold), base))

TARG = {w for w, c in swapped_in.items() if c >= 2}
print('tamper-target vocab (count>=2):', len(TARG))


def flags(df):
    out = []
    for T, C in zip(df['T'], df['C']):
        cw = tok(C); cl = {w.lower() for w in cw}
        cents = {w.lower() for w in cw if w.lower() in ENT}
        ws = tok(T)
        f = dict(miss=0, miss_targ=0, slot=0, slot_targ=0, n_ent=0)
        for i, w in enumerate(ws):
            wl = w.lower()
            if wl not in ENT:
                continue
            f['n_ent'] += 1
            if wl in cl:
                continue
            f['miss'] = 1
            if wl in TARG:
                f['miss_targ'] = 1
            # slot competitor: body has an entity that shares a preceding word
            comp = set(COMPAT.get(wl, ()))
            if i > 0:
                comp |= prev2ent.get(ws[i - 1].lower(), set())
            if (comp & cents) - {wl}:
                f['slot'] = 1
                if wl in TARG:
                    f['slot_targ'] = 1
        out.append(f)
    return pd.DataFrame(out, index=df.index)


F = flags(cold)
print('\n--- P(label=0 | flag) on cold rows ---')
for c in ['miss', 'miss_targ', 'slot', 'slot_targ']:
    m = F[c].values == 1
    if m.sum() == 0:
        print('  %-10s never fires' % c); continue
    p0 = 1 - y[m].mean()
    rec = ((y == 0) & m).sum() / (y == 0).sum()
    print('  %-10s fires %5d (%.1f%%)  P(0)=%.3f  lift=%.2fx  recall0=%.3f'
          % (c, m.sum(), 100 * m.mean(), p0, p0 / base, rec))

# best single-rule Macro F1 on cold rows
print('\n--- best single-rule cold Macro F1 ---')
for c in ['miss', 'miss_targ', 'slot', 'slot_targ']:
    pred = np.where(F[c].values == 1, 0, 1)
    print('  %-10s %.4f' % (c, f1_score(y, pred, average='macro')))
print('  all-ones   %.4f' % f1_score(y, np.ones_like(y), average='macro'))
