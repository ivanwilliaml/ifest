"""Priority diagnostics for A_cold F1-0=0.34 bottleneck (① error taxonomy FN/FP,
② similarity-feature distribution class0 vs class1, ③ most-confident FN examples).
No new model architecture -- reuses the same base41+struct7 pipeline, grouped 5-fold OOF.
Saves oof.npy + features.csv + train_slim.csv so further diagnostics don't need to
recompute the expensive LSA/BM25/entity-mining stage.
"""
import os, sys, re, time, math, hashlib, unicodedata, string as _s
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
try:
    import lightgbm as lgb; HAS_LGB=True
except Exception:
    HAS_LGB=False
    from sklearn.ensemble import HistGradientBoostingClassifier

SEED=42
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

CACHE_DIR = r"D:\Lomba\IFEST2026_DAC\notebook_v12\_diag_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

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
ANTONYMS=[('naik','turun'),('mulai','tunda'),('setuju','tolak'),('dukung','lawan'),
          ('izinkan','larang'),('terima','tolak'),('buka','tutup'),('lanjut','hentikan'),
          ('terapkan','tunda'),('naikkan','turunkan'),('tambah','kurang')]
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
        neg_t=any(w in tset for w in NEG); neg_c=any(w in cset for w in NEG)
        action_hit=0.0
        for w1,w2 in ANTONYMS:
            if (w1 in tset and w2 in cset) or (w2 in tset and w1 in cset): action_hit=1.0; break
        rows.append(dict(
            n_shared=len(inter), jaccard=len(inter)/max(len(union),1), dice=2*len(inter)/max(len(tset)+len(cset),1),
            title_cov=len(inter)/max(len(tset),1), content_cov=len(inter)/max(len(cset),1),
            bigram_ov=len(tb&cb)/max(len(tb|cb),1), idf_title_cov=num_/max(den_,1e-6),
            n_title_ent=len(tents), n_title_ent_missing=len(miss), frac_title_ent_missing=len(miss)/max(len(tents),1),
            any_title_ent_missing=float(len(miss)>0), swap_signature=float(sig>0), n_swap_sig=sig,
            n_num_title=len(tn), n_num_missing=len(tn-cn), num_overlap=len(tn&cn)/max(len(tn),1) if tn else 1.0,
            num_mismatch=float(bool(tn) and not (tn&cn)), n_year_title=len(ty), year_missing=float(bool(ty) and not (ty&cy)),
            neg_title=float(neg_t), neg_body=float(neg_c), neg_mismatch=float(neg_t!=neg_c),
            action_conflict=action_hit,
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
FEATS = list(Ftr.columns)
log(f"features: {Ftr.shape}")

_cnt=train['ch'].value_counts()
cold_mask=train['ch'].map(_cnt).eq(1).values
log(f"cold rows: {cold_mask.sum()}")

groups=train['ch'].values
sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
folds=list(sgkf.split(train,y,groups))
oof=np.zeros(len(train))
fold_id=np.zeros(len(train),dtype=int)
for i,(a,b) in enumerate(folds):
    Xa=Ftr.iloc[a].values; Xb=Ftr.iloc[b].values
    if HAS_LGB:
        m=lgb.LGBMClassifier(n_estimators=600,learning_rate=0.05,num_leaves=31,subsample=0.9,
                              colsample_bytree=0.8,class_weight='balanced',random_state=SEED,verbose=-1)
    else:
        m=HistGradientBoostingClassifier(max_iter=500,learning_rate=0.05,random_state=SEED)
    m.fit(Xa,y[a])
    oof[b]=m.predict_proba(Xb)[:,1]
    fold_id[b]=i
    log(f"  fold{i} done")

# save cache for later diagnostics (questions 4-15) without recomputing
np.save(os.path.join(CACHE_DIR,'oof.npy'), oof)
np.save(os.path.join(CACHE_DIR,'cold_mask.npy'), cold_mask)
np.save(os.path.join(CACHE_DIR,'fold_id.npy'), fold_id)
Ftr.to_csv(os.path.join(CACHE_DIR,'features.csv'), index=False)
train[['id','title','content','label']].to_csv(os.path.join(CACHE_DIR,'train_slim.csv'), index=False)
log("cache saved to _diag_cache/")

# best threshold for cold subset (macro F1)
cold_idx = np.where(cold_mask)[0]
cold_oof = oof[cold_idx]; cold_y = y[cold_idx]
best=(0.5,-1)
for t in np.arange(0.05,0.96,0.025):
    mac=f1_score(cold_y,(cold_oof>=t).astype(int),average='macro')
    if mac>best[1]: best=(t,mac)
THR=best[0]
pred_cold = (cold_oof>=THR).astype(int)
f1c = f1_score(cold_y,pred_cold,average=None,labels=[0,1])
print(f"chosen threshold={THR:.3f}  macroF1={f1c.mean():.4f}  F1_0={f1c[0]:.4f}  F1_1={f1c[1]:.4f}")

FN_mask = (cold_y==0) & (pred_cold==1)   # actual 0, predicted 1
FP_mask = (cold_y==1) & (pred_cold==0)   # actual 1, predicted 0
print(f"\nFalse Negatives (actual=0, predicted=1): {FN_mask.sum()}")
print(f"False Positives (actual=1, predicted=0): {FP_mask.sum()}")

FN_idx_global = cold_idx[FN_mask]
FP_idx_global = cold_idx[FP_mask]

def taxonomy(idx_list, name):
    print(f"\n=== {name} error taxonomy (n={len(idx_list)}) ===")
    cats = Counter()
    for idx in idx_list:
        f = Ftr.iloc[idx]
        tagged=False
        if f['any_title_ent_missing']==1.0: cats['entity_missing']+=1; tagged=True
        if f['num_mismatch']==1.0: cats['number_mismatch']+=1; tagged=True
        if f['year_missing']==1.0: cats['date_year_mismatch']+=1; tagged=True
        if f['neg_mismatch']==1.0: cats['negation_mismatch']+=1; tagged=True
        if f['action_conflict']==1.0: cats['action_antonym_conflict']+=1; tagged=True
        if not tagged:
            if f['jaccard']<0.05: cats['low_lexical_overlap_other']+=1
            else: cats['semantic_paraphrase_or_other']+=1
    total=len(idx_list)
    for k,v in cats.most_common():
        print(f"  {k:30s} n={v:4d}  ({v/total*100:5.1f}%)")
    uncat = total - sum(cats.values())
    return cats

fn_cats = taxonomy(FN_idx_global, "FALSE NEGATIVE (real fake headline, model said OK)")
fp_cats = taxonomy(FP_idx_global, "FALSE POSITIVE (real valid headline, model said fake)")

log("=== ② distribution comparison: class0 vs class1 (A_cold) ===")
sim_feats = ['jaccard','bm25_max','bm25_mean','lsa_cos','struct_max_chunk','struct_std','struct_topk_mean','title_cov']
c0 = Ftr.iloc[cold_idx][cold_y==0]
c1 = Ftr.iloc[cold_idx][cold_y==1]
rows=[]
for f in sim_feats:
    a=c0[f].values; b=c1[f].values
    rows.append(dict(feature=f,
        c0_mean=a.mean(), c0_median=np.median(a), c0_p10=np.quantile(a,.1), c0_p90=np.quantile(a,.9),
        c1_mean=b.mean(), c1_median=np.median(b), c1_p10=np.quantile(b,.1), c1_p90=np.quantile(b,.9),
        mean_gap=b.mean()-a.mean(),
        auc = roc_auc_score(cold_y, Ftr.iloc[cold_idx][f].values*(-1 if b.mean()>a.mean() else 1))
    ))
distdf=pd.DataFrame(rows)
print(distdf.round(4).to_string(index=False))

log("=== ③ top 30 most-confident FN (highest p(class1) among actual=0) ===")
fn_order = np.argsort(-cold_oof[FN_mask]) if FN_mask.sum()>0 else []
fn_sorted_idx = FN_idx_global[np.argsort(-oof[FN_idx_global])]
top_fn = fn_sorted_idx[:30]
recs=[]
for idx in top_fn:
    f=Ftr.iloc[idx]
    reasons=[]
    if f['any_title_ent_missing']==1.0: reasons.append('entity_missing')
    if f['num_mismatch']==1.0: reasons.append('number_mismatch')
    if f['year_missing']==1.0: reasons.append('year_mismatch')
    if f['neg_mismatch']==1.0: reasons.append('negation_mismatch')
    if f['action_conflict']==1.0: reasons.append('action_conflict')
    if not reasons: reasons.append('no_heuristic_flag(likely semantic)')
    recs.append(dict(
        title=train['title'].iloc[idx][:90],
        content_snippet=train['content'].iloc[idx][:180],
        true_label=0, pred_confidence_class1=round(float(oof[idx]),4),
        heuristic_reason=','.join(reasons),
    ))
fn_df=pd.DataFrame(recs)
pd.set_option('display.max_colwidth',70)
print(fn_df.to_string(index=False))
fn_df.to_csv(os.path.join(CACHE_DIR,'top30_confident_FN.csv'),index=False)

log(f"total runtime: {(time.time()-T0)/60:.1f} min")
