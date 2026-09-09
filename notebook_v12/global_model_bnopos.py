"""Global headline-body model (v12 base+struct7 features) + historical-memory features
+ retrieval score, trained on ALL train rows via a random 85/15 split (matching the
project's established regime-B validation methodology -- GroupKFold-by-body would put
every regime-B/C row's siblings in the SAME fold, making memory features always zero).
Evaluated per-regime (A_cold / B_has_pos / B_no_pos / C_exact) on the held-out 15%.
"""
import os, re, time, math, hashlib, unicodedata, string as _s
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
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
train['T']=train['title'].apply(clean); train['C']=train['content'].apply(clean)
y=train['label'].values

log("entity mining...")
up,lo=Counter(),Counter()
for t in train['C']:
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

def toks(s): return re.findall(r"[\w']+", s.lower())

log("chunk index (BM25) + LSA...")
def chunks_of(s,w=50,st=25):
    ws=s.split()
    if not ws: return ['']
    if len(ws)<=w: return [' '.join(ws)]
    out=[]
    for i in range(0,len(ws),st):
        c=ws[i:i+w]
        if not c: break
        out.append(' '.join(c))
        if i+w>=len(ws): break
    return out
allb=train[['ch','C']].drop_duplicates('ch')
chunk_txt={h:chunks_of(c) for h,c in zip(allb['ch'],allb['C'])}
df_cnt=Counter(); total_len=0; n_chunks=0; chunk_tf={}
for h,cs in chunk_txt.items():
    tfs=[]
    for c in cs:
        tk=toks(c); tf=Counter(tk); tfs.append((tf,len(tk)))
        total_len+=len(tk); n_chunks+=1
        for t in tf: df_cnt[t]+=1
    chunk_tf[h]=tfs
AVGDL=total_len/max(n_chunks,1)
IDF={t: math.log(1+(n_chunks-c+0.5)/(c+0.5)) for t,c in df_cnt.items()}
k1,b=1.5,0.75
def bm25_scores(title, ch):
    q=toks(title); out=[]
    for tf,L in chunk_tf[ch]:
        s=0.0
        for t in q:
            f=tf.get(t,0)
            if f: s+=IDF.get(t,0.0)*(f*(k1+1))/(f+k1*(1-b+b*L/AVGDL))
        out.append(s)
    return np.array(out) if out else np.array([0.0])

corpus = pd.concat([train['T'],train['C']]).tolist()
lsa_tfidf=TfidfVectorizer(ngram_range=(1,1),min_df=3,sublinear_tf=True,max_features=80000)
Xc=lsa_tfidf.fit_transform(corpus)
svd=TruncatedSVD(n_components=160,random_state=SEED); svd.fit(Xc)
def embed(texts): return normalize(svd.transform(lsa_tfidf.transform(texts)))
d_t=embed(train['T'].tolist()); d_c=embed(train['C'].tolist())
train['lsa_cos']=np.sum(d_t*d_c,axis=1)

chunk_owner=[]; flat_chunks=[]
for h,cs in chunk_txt.items():
    for c in cs: chunk_owner.append(h); flat_chunks.append(c)
CHUNK_VEC=embed(flat_chunks)
chunk_vecs=defaultdict(list)
for h,v in zip(chunk_owner,CHUNK_VEC): chunk_vecs[h].append(v)
for h in list(chunk_vecs.keys()): chunk_vecs[h]=np.array(chunk_vecs[h])

log("base41 + struct7 features (v12 pipeline)...")
def build_base(df):
    rows=[]
    tvecs_all = embed(df['T'].tolist())
    for T,C,ch,lsa,tv in zip(df['T'],df['C'],df['ch'],df['lsa_cos'],tvecs_all):
        tw=toks(T); cw=toks(C)
        _craw=[w.strip(_s.punctuation) for w in C.split()]
        _n=max(len(_craw),1); _first={}; _tf={}
        for _i,_w in enumerate(_craw):
            _tf[_w]=_tf.get(_w,0)+1
            if _w not in _first: _first[_w]=_i
        _bmax=max([_tf[e] for e in _tf if e in ENT], default=1)
        _tents=[w for w in T.split() if w in ENT]
        _pres=[e for e in _tents if e in _first]
        if _pres:
            _fp=[_first[e]/_n for e in _pres]; _rf=[_tf[e]/max(_bmax,1) for e in _pres]
            sal_maxfirst=max(_fp); sal_meanfirst=sum(_fp)/len(_fp); sal_minrelfreq=min(_rf)
            sal_late=float(max(_fp)>0.5); sal_out_of_lead=float(max(_first[e] for e in _pres)>60)
            sal_singleton=float(min(_tf[e] for e in _pres)==1)
            sal_lead_frac=sum(1 for e in _pres if _first[e]<=60)/len(_pres)
        else:
            sal_maxfirst=sal_meanfirst=1.0; sal_minrelfreq=0.0
            sal_late=sal_out_of_lead=sal_singleton=1.0; sal_lead_frac=0.0
        sal_n_present=float(len(_pres)); sal_all_present=float(bool(_tents) and len(_pres)==len(_tents))
        tset,cset=set(tw),set(cw)
        craw={w.strip(_s.punctuation) for w in C.split()}
        inter=tset&cset; union=tset|cset
        tb=set(zip(tw,tw[1:])); cb=set(zip(cw,cw[1:]))
        tents=[w for w in T.split() if w in ENT]; miss=[w for w in tents if w not in craw]
        sig=0; Ttok=T.split()
        for i,w in enumerate(Ttok):
            if w in ENT and w not in craw:
                nb=[Ttok[j] for j in (i-1,i+1) if 0<=j<len(Ttok)]
                nb=[x for x in nb if x.lower() not in ('dan','di','ke','yang')]
                if nb and all(x.strip(_s.punctuation) in craw for x in nb): sig+=1
        tn,cn=set(NUM.findall(T)),set(NUM.findall(C)); ty,cy=set(YEAR.findall(T)),set(YEAR.findall(C))
        bm=bm25_scores(T,ch); order=np.argsort(-bm); topk=bm[order[:5]]
        num_=sum(IDF.get(t,0.0) for t in tset&cset); den_=sum(IDF.get(t,0.0) for t in tset)
        svs=chunk_vecs.get(ch)
        if svs is None or len(svs)==0:
            s_first=s_first3=s_max=s_topk=s_std=s_min=s_argmax=0.0
        else:
            sims=svs@tv; kk=min(5,len(sims)); tkm=np.sort(sims)[-kk:]
            s_first=float(sims[0]); s_first3=float(sims[:min(3,len(sims))].mean())
            s_max=float(sims.max()); s_topk=float(tkm.mean()); s_std=float(sims.std())
            s_min=float(sims.min()); s_argmax=float(np.argmax(sims)/max(len(sims)-1,1))
        rows.append(dict(
            n_shared=len(inter), jaccard=len(inter)/max(len(union),1), dice=2*len(inter)/max(len(tset)+len(cset),1),
            title_cov=len(inter)/max(len(tset),1), content_cov=len(inter)/max(len(cset),1),
            bigram_ov=len(tb&cb)/max(len(tb|cb),1), idf_title_cov=num_/max(den_,1e-6),
            n_title_ent=len(tents), n_title_ent_missing=len(miss), frac_title_ent_missing=len(miss)/max(len(tents),1),
            any_title_ent_missing=float(len(miss)>0), swap_signature=float(sig>0), n_swap_sig=sig,
            n_num_title=len(tn), n_num_missing=len(tn-cn), num_overlap=len(tn&cn)/max(len(tn),1) if tn else 1.0,
            num_mismatch=float(bool(tn) and not (tn&cn)), n_year_title=len(ty), year_missing=float(bool(ty) and not (ty&cy)),
            neg_title=float(any(w in tset for w in NEG)), neg_body=float(any(w in cset for w in NEG)),
            neg_mismatch=float(any(w in tset for w in NEG)!=any(w in cset for w in NEG)),
            bm25_max=float(bm.max()), bm25_mean=float(bm.mean()), bm25_top=float(topk.mean()), bm25_std=float(bm.std()),
            bm25_argmax_pos=float(order[0]/max(len(bm)-1,1)), n_chunks=len(bm), lsa_cos=float(lsa),
            len_title=len(tw), len_content=len(cw), len_ratio=len(tw)/max(len(cw),1),
            sal_maxfirst=sal_maxfirst, sal_meanfirst=sal_meanfirst, sal_minrelfreq=sal_minrelfreq, sal_late=sal_late,
            sal_out_of_lead=sal_out_of_lead, sal_singleton=sal_singleton, sal_lead_frac=sal_lead_frac,
            sal_n_present=sal_n_present, sal_all_present=sal_all_present,
            struct_first_chunk=s_first, struct_first3_mean=s_first3, struct_max_chunk=s_max,
            struct_topk_mean=s_topk, struct_std=s_std, struct_min_chunk=s_min, struct_argmax_pos=s_argmax,
        ))
    return pd.DataFrame(rows,index=df.index)
Ftr = build_base(train)
log(f"base+struct feature matrix: {Ftr.shape}")

# ---- title TF-IDF for the retrieval-score auxiliary feature (max_char, best from bnopos_retrieval_v2) ----
char_tv = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=1, sublinear_tf=True, max_features=20000)
Cm = char_tv.fit_transform(train['T'])
Cm = Cm.multiply(1.0/np.maximum(np.sqrt(Cm.multiply(Cm).sum(axis=1)),1e-9)).tocsr()

def build_lookups(idx_pool):
    pair={}; byc=defaultdict(list)
    for i in idx_pool:
        th,ch,lb = train['th'].iloc[i], train['ch'].iloc[i], y[i]
        pair[(th,ch)]=lb; byc[ch].append((i,lb))
    return pair,byc

def regime_and_memory_features(idx_all, pair, byc):
    """For each row in idx_all, compute regime label + historical-memory features using
    ONLY entries in `byc` (built from the train-part of the split, or leave-self-out)."""
    reg=[]; hist_feats=[]
    for i in idx_all:
        th, ch = train['th'].iloc[i], train['ch'].iloc[i]
        entries = [(j,l) for j,l in byc.get(ch,[]) if j != i]  # exclude self always
        n = len(entries)
        pos_n = sum(1 for _,l in entries if l==1); neg_n = n - pos_n
        pos_rate = pos_n/n if n>0 else -1.0  # -1 sentinel = no history at all
        # title diversity: mean pairwise char-sim among historical titles for this body
        if n >= 2:
            idxs=[j for j,_ in entries]
            sims=[]
            for a in range(len(idxs)):
                for c in range(a+1,len(idxs)):
                    sims.append(float(Cm[idxs[a]].multiply(Cm[idxs[c]]).sum()))
            diversity = 1.0 - float(np.mean(sims))
        else:
            diversity = -1.0
        # retrieval score vs negative historical titles only (max_char)
        neg_idx=[j for j,l in entries if l==0]
        if neg_idx:
            retr = float(max(float(Cm[i].multiply(Cm[j]).sum()) for j in neg_idx))
        else:
            retr = -1.0
        if (th,ch) in pair: r='C_exact'
        elif n>0 and pos_n>0: r='B_has_pos'
        elif n>0: r='B_no_pos'
        else: r='A_cold'
        reg.append(r)
        hist_feats.append(dict(hist_pair_count=n, hist_pos_count=pos_n, hist_neg_count=neg_n,
                                hist_pos_rate=pos_rate, title_diversity=diversity, retrieval_max_char=retr))
    return np.array(reg), pd.DataFrame(hist_feats, index=idx_all)

log("=== random 85/15 split (matches established regime-B validation methodology) ===")
rs=np.random.RandomState(SEED); perm=rs.permutation(len(train)); cut=int(0.85*len(train))
tr_i, va_i = perm[:cut], perm[cut:]
pair_tr, byc_tr = build_lookups(tr_i)
reg_tr, hist_tr = regime_and_memory_features(tr_i, pair_tr, byc_tr)
reg_va, hist_va = regime_and_memory_features(va_i, pair_tr, byc_tr)  # memory from TRAIN part only
log(f"train regimes: {pd.Series(reg_tr).value_counts().to_dict()}")
log(f"val   regimes: {pd.Series(reg_va).value_counts().to_dict()}")

Xtr = pd.concat([Ftr.iloc[tr_i].reset_index(drop=True), hist_tr.reset_index(drop=True)], axis=1)
Xva = pd.concat([Ftr.iloc[va_i].reset_index(drop=True), hist_va.reset_index(drop=True)], axis=1)
ytr, yva = y[tr_i], y[va_i]

log("=== training GLOBAL model (all train rows, base+struct+historical-memory features) ===")
if HAS_LGB:
    m=lgb.LGBMClassifier(n_estimators=600,learning_rate=0.05,num_leaves=31,subsample=0.9,
                          colsample_bytree=0.8,class_weight='balanced',random_state=SEED,verbose=-1)
else:
    m=HistGradientBoostingClassifier(max_iter=500,learning_rate=0.05,random_state=SEED)
m.fit(Xtr.values, ytr)
proba = m.predict_proba(Xva.values)[:,1]

def eval_at(mask, t, name, results):
    if mask.sum()==0: return
    yy=yva[mask]; pp=(proba[mask]>=t).astype(int)
    acc=accuracy_score(yy,pp); f1c=f1_score(yy,pp,average=None,labels=[0,1]); mac=f1c.mean()
    print(f"{name:30s} n={mask.sum():4d} acc={acc:.4f} macroF1={mac:.4f} F1_0={f1c[0]:.4f} F1_1={f1c[1]:.4f}")
    results.append(dict(name=name,n=int(mask.sum()),acc=acc,macro_f1=mac,f1_0=f1c[0],f1_1=f1c[1]))

log("=== threshold sweep (global, all val rows) ===")
best=(0.5,-1)
for t in np.arange(0.05,0.96,0.025):
    mac=f1_score(yva,(proba>=t).astype(int),average='macro')
    if mac>best[1]: best=(t,mac)
THR=best[0]
print(f"chosen global threshold: {THR:.3f}  (macro F1 on ALL val = {best[1]:.4f})")

log("=== per-regime breakdown at chosen global threshold ===")
results=[]
eval_at(np.ones(len(yva),bool), THR, "ALL val rows", results)
for r in ['A_cold','B_has_pos','B_no_pos','C_exact']:
    eval_at(reg_va==r, THR, r, results)

log("=== compare: global-model B_no_pos vs previous specialist attempts ===")
print("  hard rule (always 1):        macroF1=0.4881")
print("  B_no_pos-only LightGBM (B3): macroF1=0.8342 (in-sample threshold -- likely inflated)")
print("  retrieval-only honest CV:    macroF1=0.5755 (max_char, repeated CV)")
bnopos_mask = reg_va=='B_no_pos'
if bnopos_mask.sum()>0:
    best_b=(0.5,-1)
    for t in np.arange(0.05,0.96,0.025):
        mac=f1_score(yva[bnopos_mask],(proba[bnopos_mask]>=t).astype(int),average='macro')
        if mac>best_b[1]: best_b=(t,mac)
    print(f"  global model, regime-specific threshold: t={best_b[0]:.3f} macroF1={best_b[1]:.4f}")

rdf=pd.DataFrame(results)
print("\n=== SUMMARY TABLE ===")
print(rdf.round(4).to_string(index=False))
rdf.to_csv('global_model_bnopos_results.csv',index=False)
log(f"total runtime: {(time.time()-T0)/60:.1f} min")
