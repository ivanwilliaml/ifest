"""Regime-B (same-body, different-title) title retrieval vs the hard rule.
Leave-one-title-out per body: for each row whose body has >=2 distinct titles in train,
the memory used to predict it is every OTHER title of that same body (real given labels,
row itself excluded) -- never its own label. This mirrors exactly what the deployed
B_has_pos/B_no_pos rule already does, just replacing "always predict 0/1" with a learned
retrieval+conflict classifier.
"""
import os, re, time, math, hashlib, unicodedata, string as _s
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold
try:
    import lightgbm as lgb; HAS_LGB=True
except Exception:
    HAS_LGB=False
    from sklearn.ensemble import HistGradientBoostingClassifier

SEED=42
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

train = pd.read_csv(r"D:\Lomba\IFEST2026_DAC\data\train.csv")
def nh(x):
    x=unicodedata.normalize('NFKC',str(x)).lower(); return re.sub(r'\s+',' ',x).strip()
train['ch']=train['content'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
train['th']=train['title'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
URL=re.compile(r'https?://\S+|www\.\S+'); HTML=re.compile(r'<[^>]+>'); WS=re.compile(r'\s+')
def clean(s):
    s=unicodedata.normalize('NFKC',str(s)); s=HTML.sub(' ',s); s=URL.sub(' ',s)
    return WS.sub(' ',s).strip()
train['T']=train['title'].apply(clean)

# ---- entity pool (same mining as v11/v12, needed for entity-conflict feature) ----
up,lo=Counter(),Counter()
for t in train['content'].apply(clean):
    for w in t.split():
        c=w.strip(_s.punctuation)
        if not c.isalpha() or len(c)<3: continue
        if c[:1].isupper(): up[c]+=1
        else: lo[c.capitalize()]+=1
pur={w:up[w]/(up[w]+lo.get(w,0)) for w in up if up[w]>=8}
tdf=Counter()
for t in train['T']:
    for w in set(t.split()): tdf[w]+=1
ENT={w for w,p in pur.items() if p>=0.95 and tdf.get(w,0)/len(train)<0.02}
NUM=re.compile(r'\d+[.,]?\d*\s*%?'); YEAR=re.compile(r'\b(?:19|20)\d{2}\b')
NEG=['tidak','bukan','belum','tanpa','gagal','ditolak','membantah','bantah','sangkal','menyangkal','menolak']

# ---- title-level TF-IDF (word + char), fit on ALL train titles once ----
word_tv = TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True, max_features=20000)
char_tv = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=1, sublinear_tf=True, max_features=20000)
Wm = word_tv.fit_transform(train['T']); Cm = char_tv.fit_transform(train['T'])
Wm = Wm.multiply(1.0/np.maximum(np.sqrt(Wm.multiply(Wm).sum(axis=1)),1e-9)).tocsr()
Cm = Cm.multiply(1.0/np.maximum(np.sqrt(Cm.multiply(Cm).sum(axis=1)),1e-9)).tocsr()
log(f"title TF-IDF built: word={Wm.shape} char={Cm.shape}")

# ---- group rows by body ----
by_body = defaultdict(list)  # ch -> list of row indices (positional)
titles = train['T'].tolist(); labels = train['label'].values
for i, ch in enumerate(train['ch']):
    by_body[ch].append(i)

def conflict_feats(a_title, b_title):
    ta, tb = a_title.split(), b_title.split()
    sa, sb = set(ta), set(tb)
    craw_b = sb  # comparing title vs title (not body) here
    ents_a = [w for w in ta if w in ENT]
    ent_conflict = float(any(w not in sb for w in ents_a) and bool(ents_a))
    na, nb = set(NUM.findall(a_title)), set(NUM.findall(b_title))
    num_conflict = float(bool(na) and na != nb)
    ya, yb = set(YEAR.findall(a_title)), set(YEAR.findall(b_title))
    year_conflict = float(bool(ya) and ya != yb)
    neg_a = any(w in sa for w in NEG); neg_b = any(w in sb for w in NEG)
    neg_conflict = float(neg_a != neg_b)
    added = len(sb-sa); removed = len(sa-sb); shared = len(sa&sb)
    shared_ratio = shared/max(len(sa|sb),1)
    return dict(ent_conflict=ent_conflict, num_conflict=num_conflict, year_conflict=year_conflict,
                neg_conflict=neg_conflict, added=added, removed=removed, shared_ratio=shared_ratio)

def build_row_features(i, memory_idx):
    """memory_idx: indices of OTHER rows sharing this row's body (row i excluded)."""
    pos_idx = [j for j in memory_idx if labels[j]==1]
    neg_idx = [j for j in memory_idx if labels[j]==0]
    def sim_block(idxs):
        if not idxs:
            return dict(max_word=0.0, top3_word=0.0, max_char=0.0, top3_char=0.0, n=0)
        wsims = np.array([float(Wm[i].multiply(Wm[j]).sum()) for j in idxs])
        csims = np.array([float(Cm[i].multiply(Cm[j]).sum()) for j in idxs])
        order = np.argsort(-wsims)
        top3w = wsims[order[:3]].mean(); top3c = np.sort(csims)[::-1][:3].mean()
        return dict(max_word=float(wsims.max()), top3_word=float(top3w),
                    max_char=float(csims.max()), top3_char=float(top3c), n=len(idxs)), order[0] if len(idxs) else None
    pos_block, pos_best = sim_block(pos_idx) if pos_idx else (dict(max_word=0.0,top3_word=0.0,max_char=0.0,top3_char=0.0,n=0), None)
    neg_block, neg_best = sim_block(neg_idx) if neg_idx else (dict(max_word=0.0,top3_word=0.0,max_char=0.0,top3_char=0.0,n=0), None)
    feats = {f'pos_{k}':v for k,v in pos_block.items()}
    feats.update({f'neg_{k}':v for k,v in neg_block.items()})
    feats['pos_neg_margin_word'] = pos_block['max_word'] - neg_block['max_word']
    feats['pos_neg_margin_char'] = pos_block['max_char'] - neg_block['max_char']
    feats['has_pos'] = float(bool(pos_idx)); feats['has_neg'] = float(bool(neg_idx))
    # conflict vs the single best-matching candidate on each side
    if pos_best is not None:
        cf = conflict_feats(titles[i], titles[pos_idx[pos_best]])
        feats.update({f'pos_{k}':v for k,v in cf.items()})
    else:
        feats.update({f'pos_{k}':0.0 for k in ['ent_conflict','num_conflict','year_conflict','neg_conflict','added','removed','shared_ratio']})
    if neg_best is not None:
        cf = conflict_feats(titles[i], titles[neg_idx[neg_best]])
        feats.update({f'neg_{k}':v for k,v in cf.items()})
    else:
        feats.update({f'neg_{k}':0.0 for k in ['ent_conflict','num_conflict','year_conflict','neg_conflict','added','removed','shared_ratio']})
    return feats

log("collecting regime-B pool (bodies with >=2 distinct titles)...")
B_rows = []
hard_rule_pred = []
for ch, idxs in by_body.items():
    if len(idxs) < 2: continue
    distinct_titles = set(train['th'].iloc[idxs])
    if len(distinct_titles) < 2: continue  # same title repeated isn't regime B
    for i in idxs:
        B_rows.append(i)
log(f"regime-B pool size: {len(B_rows)}")

log("building features for regime-B rows (leave-one-title-out memory)...")
feat_rows = []
for k, i in enumerate(B_rows):
    ch = train['ch'].iloc[i]
    memory_idx = [j for j in by_body[ch] if j != i]
    f = build_row_features(i, memory_idx)
    f['label'] = labels[i]
    # hard rule replica: predict 0 if any OTHER title on this body is labeled 1, else 1
    f['hard_rule_pred'] = 0 if any(labels[j]==1 for j in memory_idx) else 1
    feat_rows.append(f)
    if (k+1) % 300 == 0: log(f"  {k+1}/{len(B_rows)}")
FB = pd.DataFrame(feat_rows)
y_B = FB['label'].values
hard_pred = FB['hard_rule_pred'].values
feat_cols = [c for c in FB.columns if c not in ('label','hard_rule_pred')]
log(f"feature matrix: {FB[feat_cols].shape}")

def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1c = f1_score(y_true, y_pred, average=None, labels=[0,1])
    mac = f1c.mean()
    print(f"{name:32s} acc={acc:.4f}  macroF1={mac:.4f}  F1_0={f1c[0]:.4f}  F1_1={f1c[1]:.4f}")
    return dict(name=name, acc=acc, macro_f1=mac, f1_0=f1c[0], f1_1=f1c[1])

log("=== BASELINES ===")
results = []
results.append(report("A: always predict 1", y_B, np.ones(len(y_B), int)))
results.append(report("B: current hard rule", y_B, hard_pred))

log("=== C: title-retrieval LightGBM (5-fold OOF) ===")
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(y_B))
X = FB[feat_cols].values
for tr_i, va_i in kf.split(X):
    if HAS_LGB:
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                min_child_samples=5, class_weight='balanced',
                                random_state=SEED, verbose=-1)
    else:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)
    m.fit(X[tr_i], y_B[tr_i])
    oof[va_i] = m.predict_proba(X[va_i])[:,1]
best = (0.5, -1, None)
for t in np.arange(0.05, 0.96, 0.025):
    p = (oof >= t).astype(int)
    mac = f1_score(y_B, p, average='macro')
    if mac > best[1]: best = (t, mac, p)
THR, _, C_pred = best
results.append(report(f"C: title-retrieval (t={THR:.3f})", y_B, C_pred))

log("=== D: hard rule + retrieval override on disagreements ===")
# where hard rule and retrieval disagree, trust retrieval only if its OOF prob is confident
D_pred = hard_pred.copy()
disagree = hard_pred != C_pred
conf = np.abs(oof - 0.5)
override = disagree & (conf > 0.3)
D_pred[override] = C_pred[override]
results.append(report("D: hard rule + confident override", y_B, D_pred))

log("=== breakdown by B_has_pos vs B_no_pos (per current rule definition) ===")
has_pos_mask = FB['has_pos'].values.astype(bool)
print("\n-- B_has_pos subset (n=%d) --" % has_pos_mask.sum())
report("  always-1", y_B[has_pos_mask], np.ones(has_pos_mask.sum(), int))
report("  hard rule", y_B[has_pos_mask], hard_pred[has_pos_mask])
report("  retrieval", y_B[has_pos_mask], C_pred[has_pos_mask])
print("\n-- B_no_pos subset (n=%d) --" % (~has_pos_mask).sum())
report("  always-1", y_B[~has_pos_mask], np.ones((~has_pos_mask).sum(), int))
report("  hard rule", y_B[~has_pos_mask], hard_pred[~has_pos_mask])
report("  retrieval", y_B[~has_pos_mask], C_pred[~has_pos_mask])

log("=== SUMMARY TABLE ===")
rdf = pd.DataFrame(results)
print(rdf.round(4).to_string(index=False))
rdf.to_csv('title_retrieval_results.csv', index=False)
log(f"total runtime: {(time.time()-T0)/60:.1f} min")
