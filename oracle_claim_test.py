"""
ORACLE TEST before V10.

Question: can claim-level decomposition (subject / action / object / location /
number / date / negation mismatch + unsupported-claim) separate label 0 from 1
on the COLD-START rows at all? We fit IN-SAMPLE (an oracle upper bound) --
if even that cannot reach a high F1, the information is not in the representation
and V10's claim pipeline cannot save us.
"""
import re, string, unicodedata, numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score

SEED = 42
D = r'D:\Lomba\IFEST2026_DAC\data'
train = pd.read_csv(D + r'\train.csv')
print('train', train.shape, dict(train.label.value_counts()))
TC = 'title' if 'title' in train.columns else train.columns[1]
CC = 'content' if 'content' in train.columns else train.columns[2]

PUNC = string.punctuation + '\u2018\u2019\u201c\u201d\u2013\u2014'


def norm(s):
    s = unicodedata.normalize('NFKC', str(s))
    return re.sub(r'\s+', ' ', s).strip()


train['T'] = train[TC].map(norm)
train['C'] = train[CC].map(norm)
train['ch'] = train['C'].str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True).map(hash)

# ---------------------------------------------------------------- vocab mining
tok = lambda s: [w.strip(PUNC) for w in s.split() if w.strip(PUNC)]
cap, low = Counter(), Counter()
for s in pd.concat([train['T'], train['C']]):
    for w in tok(s):
        (cap if w[:1].isupper() else low)[w.lower()] += 1
ENT = {w for w, c in cap.items() if c >= 5 and c / (c + low.get(w, 0)) >= 0.95}
print('entity vocab', len(ENT))

# Indonesian verbal morphology, derived from the competition corpus only.
VPRE = ('mem', 'men', 'meng', 'meny', 'me', 'ber', 'ter', 'di', 'per')
allw = Counter()
for s in pd.concat([train['T'], train['C']]):
    for w in tok(s):
        allw[w.lower()] += 1
ACT = {w for w, c in allw.items()
       if c >= 5 and len(w) > 5 and w.startswith(VPRE) and w not in ENT}
print('action vocab', len(ACT))

NEG = {'tidak', 'bukan', 'belum', 'tanpa', 'batal', 'gagal', 'tolak', 'menolak',
       'membantah', 'bantah', 'urung', 'tak'}
NUMRE = re.compile(r'\d')
MONTH = {'januari', 'februari', 'maret', 'april', 'mei', 'juni', 'juli', 'agustus',
         'september', 'oktober', 'november', 'desember'}

# ------------------------------------------------- mutation-type oracle (S3 ask)
mut = Counter()
n_pairs = 0
for h, g in train.groupby('ch'):
    pos = g[g.label == 1]['T'].tolist()
    neg = g[g.label == 0]['T'].tolist()
    for p in pos:
        for n in neg:
            pt, nt = p.split(), n.split()
            op = [w for w in pt if w not in nt]
            on = [w for w in nt if w not in pt]
            if len(op) == 1 and len(on) == 1:
                n_pairs += 1
                a, b = op[0].strip(PUNC), on[0].strip(PUNC)
                al, bl = a.lower(), b.lower()
                if NUMRE.search(a) or NUMRE.search(b):
                    k = 'number'
                elif al in MONTH or bl in MONTH:
                    k = 'date'
                elif al in NEG or bl in NEG:
                    k = 'negation'
                elif al in ACT or bl in ACT:
                    k = 'action'
                elif al in ENT or bl in ENT:
                    k = 'entity'
                else:
                    k = 'generic'
                mut[k] += 1
print('\n--- mutation types across %d minimal pairs ---' % n_pairs)
for k, c in mut.most_common():
    print('  %-9s %5d  %.3f' % (k, c, c / max(n_pairs, 1)))

# --------------------------------------------------------- claim decomposition
def claim_feats(df):
    rows = []
    for T, C in zip(df['T'], df['C']):
        cw = tok(C)
        cl = [w.lower() for w in cw]
        cset = set(cl)
        lead = ' '.join(C.split()[:60]).lower()
        lset = set(w.strip(PUNC) for w in lead.split())
        tw = tok(T)
        tl = [w.lower() for w in tw]

        ents = [w for w in tw if w.lower() in ENT]
        acts = [w for w in tl if w in ACT]
        nums = [w for w in tl if NUMRE.search(w)]
        dates = [w for w in tl if w in MONTH]
        negs = [w for w in tl if w in NEG]

        miss = lambda xs: sum(1 for x in xs if x.lower() not in cset)
        cov = lambda xs: (1.0 if not xs else 1 - miss(xs) / len(xs))

        # contradiction: an action in the body that is a *different* action than
        # the title's, sharing the same subject entity nearby.
        body_acts = {w for w in cl if w in ACT}
        act_conf = float(bool(acts) and bool(body_acts) and
                         not (set(acts) & body_acts))

        rows.append(dict(
            n_ent=len(ents), ent_cov=cov(ents), ent_miss=miss(ents),
            ent_miss_lead=sum(1 for x in ents if x.lower() not in lset),
            n_act=len(acts), act_cov=cov(acts), act_miss=miss(acts),
            act_conflict=act_conf,
            n_num=len(nums), num_cov=cov(nums), num_miss=miss(nums),
            n_date=len(dates), date_miss=miss(dates),
            n_neg=len(negs), neg_in_body=float(bool(set(negs) & cset)),
            neg_mismatch=float(bool(negs) != bool(set(NEG) & cset)),
            tok_cov=len([w for w in tl if w in cset]) / max(len(tl), 1),
            lead_cov=len([w for w in tl if w in lset]) / max(len(tl), 1),
            unsupported=sum(1 for w in tl if w not in cset and len(w) > 4) / max(len(tl), 1),
            len_t=len(tl), len_c=len(cl),
        ))
    return pd.DataFrame(rows, index=df.index)


sizes = train.groupby('ch').size()
cold = train[train['ch'].map(sizes) == 1].copy()
print('\ncold-start analogue rows: %d  class0=%.3f' % (len(cold), 1 - cold.label.mean()))

X = claim_feats(cold)
y = cold.label.values

# -------------------------------------------------- univariate signal per feature
print('\n--- univariate AP vs 5.3%% base rate (higher = real signal) ---')
base = 1 - y.mean()
for c in X.columns:
    v = X[c].values.astype(float)
    ap = max(average_precision_score(1 - y, v), average_precision_score(1 - y, -v))
    if ap > base * 1.5:
        print('  %-15s AP=%.4f  lift=%.2fx' % (c, ap, ap / base))


def best_f1(p, y):
    best = (0, 0)
    for t in np.arange(0.02, 0.99, 0.01):
        f = f1_score(y, (p >= t).astype(int), average='macro')
        if f > best[0]:
            best = (f, t)
    return best


# ------------------------------------------------------------------ ORACLE fits
print('\n--- ORACLE (in-sample fit = upper bound, NOT achievable out of sample) ---')
for name, mdl in [
    ('tree depth-4 ', DecisionTreeClassifier(max_depth=4, random_state=SEED)),
    ('tree depth-8 ', DecisionTreeClassifier(max_depth=8, random_state=SEED)),
    ('tree unlimited', DecisionTreeClassifier(random_state=SEED)),
    ('RF 300 in-samp', RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
]:
    mdl.fit(X, y)
    p = mdl.predict_proba(X)[:, 1]
    f, t = best_f1(p, y)
    print('  %s  cold Macro F1 = %.4f  @thr %.2f' % (name, f, t))

# ------------------- what a given cold-start quality buys at competition level
# Simulate the test regime mixture: cold rows get the model, B/C get the rules
# (assumed exact, which is what the measurements say they are).
N = 3603
MIX = {'A_cold': 0.636, 'C_exact': 0.257, 'B_has_pos': 0.060, 'B_no_pos': 0.046}
n_cold = int(N * MIX['A_cold'])


def simulate(rec0, prec0):
    """rec0/prec0 = class-0 recall/precision achieved on cold rows."""
    n0 = int(n_cold * base)            # true negatives present in cold rows
    tp0 = int(n0 * rec0)               # caught
    fp0 = int(tp0 / prec0 - tp0) if prec0 > 0 else 0
    yt, yp = [], []
    yt += [0] * n0 + [1] * (n_cold - n0)
    yp += [0] * tp0 + [1] * (n0 - tp0) + [0] * fp0 + [1] * (n_cold - n0 - fp0)
    for k in ['C_exact', 'B_has_pos', 'B_no_pos']:      # rules: assumed exact
        n = int(N * MIX[k])
        z = 0 if k == 'B_has_pos' else 1
        yt += [z] * n
        yp += [z] * n
    return f1_score(yt, yp, average='macro')


print('\n--- competition-level payoff of cold-start quality ---')
print('  rules-only (cold = all 1): %.4f' % simulate(0.0, 1.0))
for rec0, prec0 in [(0.2, 0.5), (0.4, 0.7), (0.6, 0.7), (0.6, 0.9),
                    (0.8, 0.9), (0.95, 0.95), (1.0, 1.0)]:
    print('  cold rec0=%.2f prec0=%.2f -> global %.4f' % (rec0, prec0, simulate(rec0, prec0)))
