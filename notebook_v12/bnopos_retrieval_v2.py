"""B_no_pos, retrieval-only (no classifier). Compares simple similarity-based scoring
schemes via REPEATED CV to check threshold stability, plus a negative-cluster-coherence
diagnostic. No LightGBM -- just a scalar score + threshold, since B0 already showed adding
features/model complexity hurts on this tiny sample (n=733, ~34 class-0).
"""
import re, time, hashlib, unicodedata, string as _s, math
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold

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
titles=train['T'].tolist(); labels=train['label'].values

by_body=defaultdict(list)
for i,ch in enumerate(train['ch']): by_body[ch].append(i)

def toks(s): return re.findall(r"[\w']+", s.lower())
def word_iter(s): return s.lower().split()

word_tv=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True,max_features=20000)
char_tv=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1,sublinear_tf=True,max_features=20000)
Wm=word_tv.fit_transform(train['T']); Cm=char_tv.fit_transform(train['T'])
Wm=Wm.multiply(1.0/np.maximum(np.sqrt(Wm.multiply(Wm).sum(axis=1)),1e-9)).tocsr()
Cm=Cm.multiply(1.0/np.maximum(np.sqrt(Cm.multiply(Cm).sum(axis=1)),1e-9)).tocsr()
log(f"title TF-IDF built: word={Wm.shape} char={Cm.shape}")

# title-level BM25 (titles-as-documents, corpus = ALL train titles)
df_cnt=Counter(); n_titles=len(titles); total_len=0
title_tf=[]
for t in titles:
    tk=toks(t); tf=Counter(tk); title_tf.append((tf,len(tk))); total_len+=len(tk)
    for w in tf: df_cnt[w]+=1
AVGDL=total_len/max(n_titles,1)
IDF={w: math.log(1+(n_titles-c+0.5)/(c+0.5)) for w,c in df_cnt.items()}
k1,b=1.5,0.75
def bm25(query, doc_idx):
    q=toks(query); tf,L=title_tf[doc_idx]; s=0.0
    for t in q:
        f=tf.get(t,0)
        if f: s+=IDF.get(t,0.0)*(f*(k1+1))/(f+k1*(1-b+b*L/AVGDL))
    return s

log("collecting B_no_pos pool...")
pool=[]
for ch, idxs in by_body.items():
    if len(idxs)<2: continue
    if len(set(train['th'].iloc[idxs]))<2: continue
    for i in idxs:
        memory=[j for j in idxs if j!=i]
        if any(labels[j]==1 for j in memory): continue
        neg=[j for j in memory if labels[j]==0]
        if not neg: continue
        pool.append((i,neg))
log(f"pool size: {len(pool)}")

# ---- Diagnostic: negative-cluster coherence (leave-one-out pairwise sim among negatives) ----
log("=== DIAGNOSTIC: negative-cluster coherence ===")
coh=[]
for i,neg in pool:
    if len(neg)<2: continue
    sims=[]
    for a in range(len(neg)):
        for c in range(a+1,len(neg)):
            sims.append(float(Wm[neg[a]].multiply(Wm[neg[c]]).sum()))
    coh.append(np.mean(sims))
coh=np.array(coh)
print(f"bodies with >=2 negatives: {len(coh)}")
print(f"mean pairwise negative-negative similarity: {coh.mean():.4f}  (std={coh.std():.4f})")
print(f"  quantiles: 10%={np.quantile(coh,.1):.3f} 50%={np.quantile(coh,.5):.3f} 90%={np.quantile(coh,.9):.3f}")

# ---- Build candidate scores per row ----
log("computing candidate scores...")
rows=[]
for i,neg in pool:
    wsims=np.array([float(Wm[i].multiply(Wm[j]).sum()) for j in neg])
    csims=np.array([float(Cm[i].multiply(Cm[j]).sum()) for j in neg])
    bsims=np.array([bm25(titles[i],j) for j in neg])
    bsims_norm = bsims/ (bsims.max()+1e-9) if bsims.max()>0 else bsims
    order_w=np.argsort(-wsims); order_c=np.argsort(-csims); order_b=np.argsort(-bsims)
    rank_w={neg[order_w[r]]:r for r in range(len(neg))}
    rank_c={neg[order_c[r]]:r for r in range(len(neg))}
    rank_b={neg[order_b[r]]:r for r in range(len(neg))}
    kk=60.0
    rrf=[1/(kk+rank_w[j]+1)+1/(kk+rank_c[j]+1)+1/(kk+rank_b[j]+1) for j in neg]
    # negative prototype centroid similarity (mean vector of negatives vs test vector)
    proto = Wm[neg].mean(axis=0)
    proto = np.asarray(proto).ravel()
    proto_norm = proto/ (np.linalg.norm(proto)+1e-9)
    test_vec = np.asarray(Wm[i].todense()).ravel()
    proto_sim = float(np.dot(test_vec, proto_norm))
    top2w = np.sort(wsims)[::-1][:2].mean(); top3w = np.sort(wsims)[::-1][:3].mean()
    rows.append(dict(
        label=labels[i],
        max_word=float(wsims.max()), max_char=float(csims.max()), max_bm25=float(bsims_norm.max()),
        top2_word=float(top2w), top3_word=float(top3w),
        rrf_max=float(max(rrf)),
        proto_sim=proto_sim,
        combo_avg=float((wsims.max()+csims.max()+bsims_norm.max())/3.0),
    ))
FB=pd.DataFrame(rows)
y=FB['label'].values
log(f"scored rows: {len(FB)}  class0={int((y==0).sum())}")

def repeated_cv_threshold(score_col, n_repeats=5, n_splits=5):
    """For each repeat/fold: pick threshold on 'train' part maximizing macro F1,
    evaluate on 'val' part. Returns per-fold val metrics + chosen thresholds."""
    s = FB[score_col].values
    records=[]
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED+rep)
        for tr_i, va_i in kf.split(s):
            best=(0.5,-1)
            for t in np.arange(s.min(), s.max(), (s.max()-s.min())/60 + 1e-9):
                p=(s[tr_i]>=t).astype(int)
                mac=f1_score(y[tr_i],p,average='macro')
                if mac>best[1]: best=(t,mac)
            thr=best[0]
            pv=(s[va_i]>=thr).astype(int)
            f1c=f1_score(y[va_i],pv,average=None,labels=[0,1])
            records.append(dict(rep=rep, threshold=thr, macro_f1=f1c.mean(), f1_0=f1c[0], f1_1=f1c[1]))
    return pd.DataFrame(records)

log("=== B0.1-B0.4: comparing simple scoring schemes (repeated 5x5 CV) ===")
schemes = ['max_word','max_char','max_bm25','top2_word','top3_word','rrf_max','proto_sim','combo_avg']
summary=[]
for sc in schemes:
    rdf = repeated_cv_threshold(sc)
    summary.append(dict(
        scheme=sc,
        mean_macro=rdf.macro_f1.mean(), std_macro=rdf.macro_f1.std(),
        mean_f1_0=rdf.f1_0.mean(), std_f1_0=rdf.f1_0.std(),
        mean_f1_1=rdf.f1_1.mean(),
        mean_thr=rdf.threshold.mean(), std_thr=rdf.threshold.std(),
    ))
    print(f"{sc:12s} macroF1={rdf.macro_f1.mean():.4f}(+/-{rdf.macro_f1.std():.4f})  "
          f"F1_0={rdf.f1_0.mean():.4f}(+/-{rdf.f1_0.std():.4f})  F1_1={rdf.f1_1.mean():.4f}  "
          f"thr={rdf.threshold.mean():.4f}(+/-{rdf.threshold.std():.4f})")

log("=== full-sample (single best threshold, whole pool) reference, for comparison to B0's 0.9167 ===")
for sc in schemes:
    s=FB[sc].values
    best=(0.5,-1)
    for t in np.arange(s.min(),s.max(),(s.max()-s.min())/100+1e-9):
        p=(s>=t).astype(int); mac=f1_score(y,p,average='macro')
        if mac>best[1]: best=(t,mac)
    p=(s>=best[0]).astype(int); f1c=f1_score(y,p,average=None,labels=[0,1])
    print(f"{sc:12s} (in-sample best t={best[0]:.4f}) macro={f1c.mean():.4f} F1_0={f1c[0]:.4f} F1_1={f1c[1]:.4f}")

summ_df=pd.DataFrame(summary).sort_values('mean_macro',ascending=False)
print("\n=== SUMMARY (sorted by mean CV macro F1, honest out-of-fold) ===")
print(summ_df.round(4).to_string(index=False))
summ_df.to_csv('bnopos_retrieval_v2_summary.csv',index=False)
log(f"total runtime: {(time.time()-T0)/60:.1f} min")
