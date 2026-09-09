"""Rebuild the A_cold dense model (base41+struct7, grouped 5-fold OOF, same as v12) and
report the 20 rows with lowest prediction confidence (probability closest to 0.5) among
genuinely cold-start rows.
"""
import re, time, math, hashlib, unicodedata, string as _s
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
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

log("chunk index + LSA...")
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

log("base41 + struct7 features...")
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
log(f"features: {Ftr.shape}")

_cnt=train['ch'].value_counts()
cold_mask=train['ch'].map(_cnt).eq(1).values
log(f"cold rows: {cold_mask.sum()}")

groups=train['ch'].values
sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
folds=list(sgkf.split(train,y,groups))
oof=np.zeros(len(train))
for i,(a,b) in enumerate(folds):
    Xa=Ftr.iloc[a].values; Xb=Ftr.iloc[b].values
    if HAS_LGB:
        m=lgb.LGBMClassifier(n_estimators=600,learning_rate=0.05,num_leaves=31,subsample=0.9,
                              colsample_bytree=0.8,class_weight='balanced',random_state=SEED,verbose=-1)
    else:
        m=HistGradientBoostingClassifier(max_iter=500,learning_rate=0.05,random_state=SEED)
    m.fit(Xa,y[a])
    oof[b]=m.predict_proba(Xb)[:,1]
    log(f"  fold{i} done")

log("=== 20 lowest-confidence cold predictions (|p-0.5| smallest) ===")
cold_idx = np.where(cold_mask)[0]
cold_oof = oof[cold_idx]
uncertainty = np.abs(cold_oof - 0.5)
order = np.argsort(uncertainty)[:20]
rows=[]
for rank, oi in enumerate(order):
    idx = cold_idx[oi]
    rows.append(dict(
        rank=rank+1, id=train['id'].iloc[idx] if 'id' in train.columns else idx,
        title=train['title'].iloc[idx][:80], content_snippet=train['content'].iloc[idx][:150],
        true_label=int(y[idx]), confidence_p1=round(float(oof[idx]),4),
        predicted=int(oof[idx]>=0.5),
    ))
outdf = pd.DataFrame(rows)
pd.set_option('display.max_colwidth', 60)
print(outdf.to_string(index=False))
outdf.to_csv('lowest_confidence_20.csv', index=False)
log(f"total runtime: {(time.time()-T0)/60:.1f} min")
