"""LB-blind feature-family ablation, evaluated purely on grouped CV (no Kaggle submit).
Runs locally against the real train.csv. Cold-start subset (content-hash count==1) is the
real target: A-regime rows are the model's actual job, everything else is answered by the
deterministic rule layer already established in v11/v12.
"""
import os, re, time, math, hashlib, unicodedata, warnings, random, string as _s, json
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import normalize
import scipy.sparse as sp
try:
    import lightgbm as lgb; HAS_LGB=True
except Exception:
    HAS_LGB=False
    from sklearn.ensemble import HistGradientBoostingClassifier

SEED=42; random.seed(SEED); np.random.seed(SEED)
T0=time.time()
def log(msg): print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

DATA=r"D:\Lomba\IFEST2026_DAC\data"
train=pd.read_csv(os.path.join(DATA,'train.csv'))
test=pd.read_csv(os.path.join(DATA,'test.csv'))
log(f"loaded train={train.shape} test={test.shape}")

def nh(x):
    x=unicodedata.normalize('NFKC',str(x)).lower(); return re.sub(r'\s+',' ',x).strip()
for d in (train,test):
    d['ch']=d['content'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
    d['th']=d['title'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())

URL=re.compile(r'https?://\S+|www\.\S+'); HTML=re.compile(r'<[^>]+>'); WS=re.compile(r'\s+')
def clean(s):
    s=unicodedata.normalize('NFKC',str(s)); s=HTML.sub(' ',s); s=URL.sub(' ',s)
    return WS.sub(' ',s).strip()
for d in (train,test):
    d['T']=d['title'].apply(clean); d['C']=d['content'].apply(clean)

y=train['label'].values; groups=train['ch'].values
sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
folds=list(sgkf.split(train,y,groups))
_cnt=train['ch'].value_counts()
cold_mask=train['ch'].map(_cnt).eq(1).values
log(f"cold rows: {cold_mask.sum()} / {len(train)}")

def toks(s): return re.findall(r"[\w']+", s.lower())
NUM=re.compile(r'\d+[.,]?\d*\s*%?'); YEAR=re.compile(r'\b(?:19|20)\d{2}\b')
NEG=['tidak','bukan','belum','tanpa','gagal','ditolak','membantah','bantah','sangkal','menyangkal','menolak']

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
log(f"entity pool: {len(ENT)}")

log("chunk index (BM25 + LSA)...")
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
allpairs=pd.concat([train[['ch','C']],test[['ch','C']]],ignore_index=True).drop_duplicates('ch')
chunk_txt={h:chunks_of(c) for h,c in zip(allpairs['ch'],allpairs['C'])}
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

corpus = pd.concat([train['T'],train['C'],test['T'],test['C']]).tolist()
lsa_tfidf=TfidfVectorizer(ngram_range=(1,1),min_df=3,sublinear_tf=True,max_features=80000)
Xc=lsa_tfidf.fit_transform(corpus)
svd=TruncatedSVD(n_components=160,random_state=SEED); svd.fit(Xc)
def embed(texts): return normalize(svd.transform(lsa_tfidf.transform(texts)))
for d in (train,test):
    d_t=embed(d['T'].tolist()); d_c=embed(d['C'].tolist())
    d['lsa_cos']=np.sum(d_t*d_c,axis=1)

log("NMF topic space (100d, non-negative TF-IDF)...")
nmf_tfidf=TfidfVectorizer(ngram_range=(1,1),min_df=3,max_features=40000)
Xn=nmf_tfidf.fit_transform(corpus)
nmf=NMF(n_components=60,random_state=SEED,max_iter=200,init='nndsvda')
nmf.fit(Xn)
def topic_dist(texts):
    W=nmf.transform(nmf_tfidf.transform(texts))
    W=np.clip(W,1e-9,None); W=W/W.sum(axis=1,keepdims=True)
    return W
for d in (train,test):
    d['_topic_T']=list(topic_dist(d['T'].tolist()))
    d['_topic_C']=list(topic_dist(d['C'].tolist()))

log("chunk LSA vectors (for structural + distribution features)...")
chunk_owner=[]; flat_chunks=[]
for h,cs in chunk_txt.items():
    for c in cs:
        chunk_owner.append(h); flat_chunks.append(c)
CHUNK_VEC=embed(flat_chunks)
chunk_vecs=defaultdict(list)
for h,v in zip(chunk_owner,CHUNK_VEC): chunk_vecs[h].append(v)
for h in list(chunk_vecs.keys()): chunk_vecs[h]=np.array(chunk_vecs[h])

log("base 41 features (v11/v12 consistency features)...")
def build_base(df):
    rows=[]
    for T,C,ch,lsa in zip(df['T'],df['C'],df['ch'],df['lsa_cos']):
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
        ))
    return pd.DataFrame(rows,index=df.index)
Ftr=build_base(train); Fte=build_base(test)
log(f"base features: {Ftr.shape[1]}")

log("[B] chunk-distribution features...")
def chunk_dist_features(df):
    rows=[]; tvecs=embed(df['T'].tolist())
    for tv,ch in zip(tvecs, df['ch']):
        svs=chunk_vecs.get(ch)
        if svs is None or len(svs)==0:
            rows.append(dict(cd_q10=0,cd_q25=0,cd_q50=0,cd_q75=0,cd_q90=0,cd_top1=0,cd_top3=0,cd_top5=0,
                              cd_first_q=0,cd_mid_h=0,cd_last_q=0,cd_argmax=0,cd_argmin=0,cd_slope=0,
                              cd_early_late_delta=0,cd_entropy=0,cd_n_above=0,cd_run_above=0))
            continue
        sims=svs@tv; n=len(sims); pos=np.arange(n)/max(n-1,1)
        q=np.quantile(sims,[0.1,0.25,0.5,0.75,0.9])
        srt=np.sort(sims)[::-1]
        top1=srt[0]; top3=srt[:min(3,n)].mean(); top5=srt[:min(5,n)].mean()
        fq=sims[pos<=0.25].mean() if (pos<=0.25).any() else sims.mean()
        mh=sims[(pos>0.25)&(pos<0.75)].mean() if ((pos>0.25)&(pos<0.75)).any() else sims.mean()
        lq=sims[pos>=0.75].mean() if (pos>=0.75).any() else sims.mean()
        argmax=int(np.argmax(sims))/max(n-1,1); argmin=int(np.argmin(sims))/max(n-1,1)
        slope=float(np.polyfit(pos,sims,1)[0]) if n>=2 else 0.0
        early_late=fq-lq
        p=np.clip(sims-sims.min()+1e-6,1e-9,None); p=p/p.sum()
        entropy=float(-(p*np.log(p)).sum())
        thr=0.5; above=sims>thr; n_above=float(above.sum())
        run=0; best=0
        for a in above:
            run = run+1 if a else 0
            best=max(best,run)
        rows.append(dict(cd_q10=q[0],cd_q25=q[1],cd_q50=q[2],cd_q75=q[3],cd_q90=q[4],
                          cd_top1=top1,cd_top3=top3,cd_top5=top5,cd_first_q=fq,cd_mid_h=mh,cd_last_q=lq,
                          cd_argmax=argmax,cd_argmin=argmin,cd_slope=slope,cd_early_late_delta=early_late,
                          cd_entropy=entropy,cd_n_above=n_above,cd_run_above=float(best)))
    return pd.DataFrame(rows,index=df.index)
CDtr=chunk_dist_features(train); CDte=chunk_dist_features(test)
log(f"chunk-dist features: {CDtr.shape[1]}")

log("[C] support vs conflict separation...")
def support_conflict(F):
    support = (F['title_cov']+F['content_cov']+F['jaccard']+F['dice']+F['bigram_ov']
               +(1-F['frac_title_ent_missing'])+F['num_overlap']+(1-F['year_missing'])
               +(1-F['neg_mismatch'])+F['lsa_cos'])
    conflict = (F['frac_title_ent_missing']+F['swap_signature']+F['num_mismatch']
                +F['year_missing']+F['neg_mismatch']+F['any_title_ent_missing'])
    return pd.DataFrame({'support_score':support,'conflict_score':conflict,'support_margin':support-conflict})
SCtr=support_conflict(Ftr); SCte=support_conflict(Fte)

log("[topic] NMF cosine + KL/JS divergence...")
def topic_features(df):
    rows=[]
    for tt,tc in zip(df['_topic_T'],df['_topic_C']):
        cos=float(np.dot(tt,tc)/(np.linalg.norm(tt)*np.linalg.norm(tc)+1e-9))
        kl_hb=float(np.sum(tt*np.log(tt/tc)))
        kl_bh=float(np.sum(tc*np.log(tc/tt)))
        m=0.5*(tt+tc); js=float(0.5*np.sum(tt*np.log(tt/m))+0.5*np.sum(tc*np.log(tc/m)))
        rows.append(dict(topic_cos=cos,topic_kl_hb=kl_hb,topic_kl_bh=kl_bh,topic_js=js))
    return pd.DataFrame(rows,index=df.index)
TPtr=topic_features(train); TPte=topic_features(test)
log(f"topic features: {TPtr.shape[1]}")

# ---- CV harness: dense-only LightGBM (sparse model excluded -- v12 already showed its
# blend weight is small/unstable and doesn't change which feature set wins) ----
def run_cv(Xtr_df, Xte_df, class_weight=None, verbose=False):
    oof=np.zeros(len(train))
    for i,(a,b) in enumerate(folds):
        Xa=Xtr_df.iloc[a].values; Xb=Xtr_df.iloc[b].values
        if HAS_LGB:
            cw = class_weight if class_weight is not None else 'balanced'
            m=lgb.LGBMClassifier(n_estimators=600,learning_rate=0.05,num_leaves=31,
                                  subsample=0.9,colsample_bytree=0.8,class_weight=cw,
                                  random_state=SEED,verbose=-1)
        else:
            m=HistGradientBoostingClassifier(max_iter=500,learning_rate=0.05,random_state=SEED)
        m.fit(Xa,y[a])
        oof[b]=m.predict_proba(Xb)[:,1]
        if verbose: log(f"    fold{i} done")
    return oof

def best_macro(oof, mask=None):
    yy = y[mask] if mask is not None else y
    oo = oof[mask] if mask is not None else oof
    best=(0.5,-1,None,None)
    for t in np.arange(0.05,0.96,0.025):
        p=(oo>=t).astype(int)
        f1c=f1_score(yy,p,average=None); mac=f1c.mean()
        if mac>best[1]: best=(t,mac,f1c[0],f1c[1])
    return best  # threshold, macro, f1_0, f1_1

results=[]

log("=== EXPERIMENT A: baseline v12 (base41 + struct7) ===")
STRtr = pd.DataFrame({  # reuse chunk_vecs, same 7 struct feats as v12 (subset of chunk-dist)
    'struct_first_chunk':CDtr['cd_top1']*0+0}, index=train.index)  # placeholder, real one below
def struct7(df, CD):
    # v12's 7 features are: first_chunk, first3_mean, max_chunk, topk_mean, std, min_chunk, argmax_pos
    # first_chunk/std/min not directly in CD; compute quickly from chunk_vecs again (cheap, cached vectors)
    rows=[]; tvecs=embed(df['T'].tolist())
    for tv,ch in zip(tvecs, df['ch']):
        svs=chunk_vecs.get(ch)
        if svs is None or len(svs)==0:
            rows.append(dict(struct_first_chunk=0,struct_first3_mean=0,struct_max_chunk=0,struct_topk_mean=0,
                              struct_std=0,struct_min_chunk=0,struct_argmax_pos=0)); continue
        sims=svs@tv; k=min(5,len(sims)); topk=np.sort(sims)[-k:]
        rows.append(dict(struct_first_chunk=float(sims[0]),struct_first3_mean=float(sims[:min(3,len(sims))].mean()),
                          struct_max_chunk=float(sims.max()),struct_topk_mean=float(topk.mean()),
                          struct_std=float(sims.std()),struct_min_chunk=float(sims.min()),
                          struct_argmax_pos=float(np.argmax(sims)/max(len(sims)-1,1))))
    return pd.DataFrame(rows,index=df.index)
S7tr=struct7(train,CDtr); S7te=struct7(test,CDte)

FA_tr=pd.concat([Ftr,S7tr],axis=1); FA_te=pd.concat([Fte,S7te],axis=1)
oof_A=run_cv(FA_tr,FA_te)
t,mac,f0,f1v=best_macro(oof_A); tc,macc,f0c,f1c_=best_macro(oof_A,cold_mask)
results.append(dict(experiment='A: baseline v12 (base+struct7)',n_feats=FA_tr.shape[1],
                     macro_all=mac,f1_0_all=f0,f1_1_all=f1v,
                     macro_cold=macc,f1_0_cold=f0c,f1_1_cold=f1c_))
log(f"A done: macro_all={mac:.4f} macro_cold={macc:.4f} f1_0_cold={f0c:.4f}")

log("=== EXPERIMENT B: + chunk-distribution features ===")
FB_tr=pd.concat([FA_tr,CDtr],axis=1); FB_te=pd.concat([FA_te,CDte],axis=1)
oof_B=run_cv(FB_tr,FB_te)
t,mac,f0,f1v=best_macro(oof_B); tc,macc,f0c,f1c_=best_macro(oof_B,cold_mask)
results.append(dict(experiment='B: + chunk-distribution (quantiles/topk/entropy/run)',n_feats=FB_tr.shape[1],
                     macro_all=mac,f1_0_all=f0,f1_1_all=f1v,
                     macro_cold=macc,f1_0_cold=f0c,f1_1_cold=f1c_))
log(f"B done: macro_all={mac:.4f} macro_cold={macc:.4f} f1_0_cold={f0c:.4f}")

log("=== EXPERIMENT C: + support/conflict separation ===")
FC_tr=pd.concat([FB_tr,SCtr],axis=1); FC_te=pd.concat([FB_te,SCte],axis=1)
oof_C=run_cv(FC_tr,FC_te)
t,mac,f0,f1v=best_macro(oof_C); tc,macc,f0c,f1c_=best_macro(oof_C,cold_mask)
results.append(dict(experiment='C: + support/conflict/margin scores',n_feats=FC_tr.shape[1],
                     macro_all=mac,f1_0_all=f0,f1_1_all=f1v,
                     macro_cold=macc,f1_0_cold=f0c,f1_1_cold=f1c_))
log(f"C done: macro_all={mac:.4f} macro_cold={macc:.4f} f1_0_cold={f0c:.4f}")

log("=== EXPERIMENT D: + NMF topic + KL/JS divergence ===")
FD_tr=pd.concat([FC_tr,TPtr],axis=1); FD_te=pd.concat([FC_te,TPte],axis=1)
oof_D=run_cv(FD_tr,FD_te)
t,mac,f0,f1v=best_macro(oof_D); tc,macc,f0c,f1c_=best_macro(oof_D,cold_mask)
results.append(dict(experiment='D: + NMF topic cosine + KL/JS divergence',n_feats=FD_tr.shape[1],
                     macro_all=mac,f1_0_all=f0,f1_1_all=f1v,
                     macro_cold=macc,f1_0_cold=f0c,f1_1_cold=f1c_))
log(f"D done: macro_all={mac:.4f} macro_cold={macc:.4f} f1_0_cold={f0c:.4f}")

log("=== EXPERIMENT E: class-weight sweep (cold subset, best feature set D) ===")
best_cw=None
for w0 in [1.0,1.5,2.0,3.0,4.0]:
    cw={0:w0,1:1.0}
    oof_w=run_cv(FD_tr,FD_te,class_weight=cw)
    tc,macc,f0c,f1c_=best_macro(oof_w,cold_mask)
    results.append(dict(experiment=f'E: class_weight 0:{w0}:1 (cold-tuned)',n_feats=FD_tr.shape[1],
                         macro_all=None,f1_0_all=None,f1_1_all=None,
                         macro_cold=macc,f1_0_cold=f0c,f1_1_cold=f1c_))
    log(f"  w0={w0}: macro_cold={macc:.4f} f1_0_cold={f0c:.4f} f1_1_cold={f1c_:.4f}")
    if best_cw is None or macc>best_cw[1]: best_cw=(w0,macc,oof_w)

log("=== EXPERIMENT F: threshold optimized for F1-0 alone (best class-weight model, cold subset) ===")
_,_,oof_best=best_cw
best_f0=(0.5,-1,None,None)
for t in np.arange(0.05,0.96,0.025):
    p=(oof_best[cold_mask]>=t).astype(int)
    f1c=f1_score(y[cold_mask],p,average=None)
    if f1c[0]>best_f0[1]: best_f0=(t,f1c[0],f1c[1],f1c.mean())
results.append(dict(experiment=f'F: threshold tuned for F1-0 only (t={best_f0[0]:.3f})',n_feats=FD_tr.shape[1],
                     macro_all=None,f1_0_all=None,f1_1_all=None,
                     macro_cold=best_f0[3],f1_0_cold=best_f0[1],f1_1_cold=best_f0[2]))
log(f"F done: t={best_f0[0]:.3f} f1_0={best_f0[1]:.4f} f1_1={best_f0[2]:.4f} macro={best_f0[3]:.4f}")

log("=== SUMMARY TABLE ===")
rdf=pd.DataFrame(results)
pd.set_option('display.width',160); pd.set_option('display.max_columns',20)
print(rdf.round(4).to_string(index=False))
rdf.to_csv('ablation_results.csv',index=False)
log(f"total runtime: {(time.time()-T0)/60:.1f} min")
