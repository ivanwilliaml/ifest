"""Honest (leakage-free threshold) evaluation of regime B (B_has_pos + B_no_pos combined),
using the title-retrieval + conflict features, validated via repeated CV where the
threshold is chosen on a train fold and evaluated on a separate held-out fold -- fixing
the same threshold-selection leak found in the earlier B_no_pos-only experiment.
"""
import re, time, hashlib, unicodedata, string as _s
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
HTML=re.compile(r'<[^>]+>'); URL=re.compile(r'https?://\S+|www\.\S+'); WS=re.compile(r'\s+')
def clean(s):
    s=unicodedata.normalize('NFKC',str(s)); s=HTML.sub(' ',s); s=URL.sub(' ',s)
    return WS.sub(' ',s).strip()
train['T']=train['title'].apply(clean)

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

word_tv = TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True, max_features=20000)
char_tv = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=1, sublinear_tf=True, max_features=20000)
Wm = word_tv.fit_transform(train['T']); Cm = char_tv.fit_transform(train['T'])
Wm = Wm.multiply(1.0/np.maximum(np.sqrt(Wm.multiply(Wm).sum(axis=1)),1e-9)).tocsr()
Cm = Cm.multiply(1.0/np.maximum(np.sqrt(Cm.multiply(Cm).sum(axis=1)),1e-9)).tocsr()
log("title TF-IDF built")

by_body = defaultdict(list)
titles = train['T'].tolist(); labels = train['label'].values
for i, ch in enumerate(train['ch']): by_body[ch].append(i)

def conflict_feats(a_title, b_title):
    ta, tb = a_title.split(), b_title.split()
    sa, sb = set(ta), set(tb)
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
    pos_idx = [j for j in memory_idx if labels[j]==1]
    neg_idx = [j for j in memory_idx if labels[j]==0]
    def sim_block(idxs):
        if not idxs: return dict(max_word=0.0, top3_word=0.0, max_char=0.0, top3_char=0.0, n=0), None
        wsims = np.array([float(Wm[i].multiply(Wm[j]).sum()) for j in idxs])
        csims = np.array([float(Cm[i].multiply(Cm[j]).sum()) for j in idxs])
        order = np.argsort(-wsims)
        return dict(max_word=float(wsims.max()), top3_word=float(np.sort(wsims)[::-1][:3].mean()),
                    max_char=float(csims.max()), top3_char=float(np.sort(csims)[::-1][:3].mean()),
                    n=len(idxs)), idxs[order[0]]
    pos_block, pos_best = sim_block(pos_idx)
    neg_block, neg_best = sim_block(neg_idx)
    feats = {f'pos_{k}':v for k,v in pos_block.items()}
    feats.update({f'neg_{k}':v for k,v in neg_block.items()})
    feats['pos_neg_margin_word'] = pos_block['max_word'] - neg_block['max_word']
    feats['pos_neg_margin_char'] = pos_block['max_char'] - neg_block['max_char']
    feats['has_pos'] = float(bool(pos_idx)); feats['has_neg'] = float(bool(neg_idx))
    if pos_best is not None:
        cf = conflict_feats(titles[i], titles[pos_best])
        feats.update({f'pos_{k}':v for k,v in cf.items()})
    else:
        feats.update({f'pos_{k}':0.0 for k in ['ent_conflict','num_conflict','year_conflict','neg_conflict','added','removed','shared_ratio']})
    if neg_best is not None:
        cf = conflict_feats(titles[i], titles[neg_best])
        feats.update({f'neg_{k}':v for k,v in cf.items()})
    else:
        feats.update({f'neg_{k}':0.0 for k in ['ent_conflict','num_conflict','year_conflict','neg_conflict','added','removed','shared_ratio']})
    return feats

log("collecting regime-B pool...")
B_rows = []
for ch, idxs in by_body.items():
    if len(idxs) < 2: continue
    if len(set(train['th'].iloc[idxs])) < 2: continue
    B_rows.extend(idxs)
log(f"regime-B pool: {len(B_rows)}")

feat_rows=[]
for k,i in enumerate(B_rows):
    ch = train['ch'].iloc[i]
    memory_idx = [j for j in by_body[ch] if j != i]
    f = build_row_features(i, memory_idx)
    f['label']=labels[i]
    f['has_pos_regime'] = f['has_pos']
    feat_rows.append(f)
FB = pd.DataFrame(feat_rows)
y_B = FB['label'].values
feat_cols = [c for c in FB.columns if c not in ('label','has_pos_regime')]
X = FB[feat_cols].values
log(f"feature matrix: {X.shape}, class0={int((y_B==0).sum())}, class1={int((y_B==1).sum())}")

def repeated_honest_cv(n_repeats=5, n_splits=5):
    records=[]
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED+rep)
        for tr_i, va_i in kf.split(X):
            if HAS_LGB:
                m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                        min_child_samples=5, class_weight='balanced', random_state=SEED, verbose=-1)
            else:
                m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)
            # further split tr_i into fit/threshold halves so the threshold never sees va_i's labels
            rs = np.random.RandomState(SEED+rep)
            perm = rs.permutation(len(tr_i)); half = len(tr_i)//2
            fit_i, thr_i = tr_i[perm[:half]], tr_i[perm[half:]]
            m.fit(X[fit_i], y_B[fit_i])
            p_thr = m.predict_proba(X[thr_i])[:,1]
            best=(0.5,-1)
            for t in np.arange(0.05,0.96,0.025):
                mac=f1_score(y_B[thr_i],(p_thr>=t).astype(int),average='macro')
                if mac>best[1]: best=(t,mac)
            thr=best[0]
            p_va = m.predict_proba(X[va_i])[:,1]
            pred_va = (p_va>=thr).astype(int)
            f1c = f1_score(y_B[va_i], pred_va, average=None, labels=[0,1])
            acc = accuracy_score(y_B[va_i], pred_va)
            records.append(dict(rep=rep, threshold=thr, acc=acc, macro_f1=f1c.mean(), f1_0=f1c[0], f1_1=f1c[1],
                                 has_pos_share=FB['has_pos_regime'].iloc[va_i].mean()))
    return pd.DataFrame(records)

log("=== honest repeated CV (fit/threshold/eval strictly separated) — regime B combined ===")
rdf = repeated_honest_cv()
print(f"n folds evaluated: {len(rdf)}")
print(f"macro F1: mean={rdf.macro_f1.mean():.4f} std={rdf.macro_f1.std():.4f}")
print(f"F1_0:     mean={rdf.f1_0.mean():.4f} std={rdf.f1_0.std():.4f}")
print(f"F1_1:     mean={rdf.f1_1.mean():.4f} std={rdf.f1_1.std():.4f}")
print(f"acc:      mean={rdf.acc.mean():.4f} std={rdf.acc.std():.4f}")
print(f"threshold: mean={rdf.threshold.mean():.4f} std={rdf.threshold.std():.4f}")
rdf.to_csv('regime_B_honest_results.csv', index=False)
log(f"total runtime: {(time.time()-T0)/60:.1f} min")
