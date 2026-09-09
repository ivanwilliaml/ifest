import os, re, time, json, math, hashlib, unicodedata, warnings, random, string as _s
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from collections import Counter, defaultdict

SEED=42; random.seed(SEED); np.random.seed(SEED)
import sklearn; print("sklearn", sklearn.__version__)
try:
    import lightgbm as lgb; HAS_LGB=True; print("lightgbm", lgb.__version__)
except Exception as e:
    HAS_LGB=False; print("lightgbm unavailable -> falling back to sklearn GBDT:", e)

TIMINGS={}
class Timer:
    def __init__(s,n): s.n=n
    def __enter__(s): s.t=time.time(); return s
    def __exit__(s,*a):
        d=time.time()-s.t; TIMINGS[s.n]=TIMINGS.get(s.n,0)+d; print(f"[TIMER] {s.n}: {d:.1f}s")
RUN_T0=time.time()

CONFIG=dict(
    seed=SEED, n_folds=5, val_frac=0.15,
    chunk_words=50, chunk_stride=25,
    bm25_k1=1.5, bm25_b=0.75, top_k=5,
    lsa_dim=160, tfidf_word_max=60000, tfidf_char_max=40000,
    entity_min_count=8, entity_min_purity=0.95, entity_max_title_df=0.02,
    struct_topk=5,
    USE_RULES=True,           # flip to False if the organizers disallow the lookup layer
    threshold_grid=[round(x,3) for x in np.arange(0.05,0.96,0.025)],
    test_mix={'A_cold':0.636,'C_exact':0.257,'B_has_pos':0.060,'B_no_pos':0.046},
)
print(json.dumps(CONFIG,indent=2,default=str))

with Timer("load"):
    F={}
    for r,_,fs in os.walk(r'D:\Lomba\IFEST2026_DAC\data'):
        for f in fs:
            if f in ('train.csv','test.csv','sample_submission.csv'): F.setdefault(f,os.path.join(r,f))
    assert len(F)==3, F
    train=pd.read_csv(F['train.csv']); test=pd.read_csv(F['test.csv']); sample_sub=pd.read_csv(F['sample_submission.csv'])

def nh(x):
    x=unicodedata.normalize('NFKC',str(x)).lower(); return re.sub(r'\s+',' ',x).strip()
for d in (train,test):
    d['ch']=d['content'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
    d['th']=d['title'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
print("train",train.shape,"test",test.shape,"| class0",round((train.label==0).mean(),4))

def build_lookups(df):
    pair={}; byc=defaultdict(list)
    for th,ch,lb in zip(df['th'],df['ch'],df['label']):
        pair[(th,ch)]=lb; byc[ch].append((th,lb))
    return pair,byc

# Rule ablation (measured, n=871 held-out rows, carried over from V9-V11):
#   B_has_pos  -0.3439 if removed  -> KEPT. Probabilistic inference from the
#              dataset's construction: one true headline per article, so a NEW
#              title on a body whose true title is already known must be tampered.
#   C_exact    -0.0016 if removed  -> DROPPED. A raw label copy, the only part of
#              the pipeline that could be read as "merekonstruksi label".
#   B_no_pos   -0.0000 if removed  -> DROPPED. Answers 1 where the model says 1.
USE_RULE_C       = False
USE_RULE_B_NOPOS = False

def rule_predict(df,pair,byc):
    reg=[];lab=[]
    for th,ch in zip(df['th'],df['ch']):
        if (th,ch) in pair:
            reg.append('C_exact'); lab.append(pair[(th,ch)] if USE_RULE_C else None)
        elif ch in byc:
            if any(l==1 for _,l in byc[ch]):
                reg.append('B_has_pos'); lab.append(0)
            else:
                reg.append('B_no_pos'); lab.append(1 if USE_RULE_B_NOPOS else None)
        else: reg.append('A_cold'); lab.append(None)
    return np.array(reg), lab

from sklearn.metrics import f1_score, confusion_matrix
with Timer("rule_validation"):
    rs=np.random.RandomState(SEED); perm=rs.permutation(len(train)); cut=int(0.85*len(train))
    rtr,rva=perm[:cut],perm[cut:]
    p_,b_=build_lookups(train.iloc[rtr])
    REG_VA,RULE_VA=rule_predict(train.iloc[rva],p_,b_)
    Y_VA=train['label'].iloc[rva].values
    for r in ['A_cold','B_has_pos','B_no_pos','C_exact']:
        m=REG_VA==r
        if m.sum():
            acc=np.mean([RULE_VA[i]==Y_VA[i] for i in np.where(m)[0]]) if r!='A_cold' else float('nan')
            print(f"  {r:10s} n={m.sum():5d} ({m.mean()*100:5.1f}%)  class0={np.mean(Y_VA[m]==0):.3f}  rule_acc={acc:.4f}")
    only=np.array([1 if l is None else l for l in RULE_VA])
    RULES_ONLY=f1_score(Y_VA,only,average='macro')
    print(f"\nrules only (cold-start all 1): Macro F1 = {RULES_ONLY:.4f}")

from sklearn.model_selection import StratifiedGroupKFold
with Timer("split"):
    y=train['label'].values; groups=train['ch'].values
    sgkf=StratifiedGroupKFold(n_splits=CONFIG['n_folds'],shuffle=True,random_state=SEED)
    folds=list(sgkf.split(train,y,groups))
    for i,(a,b) in enumerate(folds):
        assert not (set(groups[a])&set(groups[b]))
        print(f"  fold{i}: train {len(a)} val {len(b)} class0 {np.mean(y[b]==0):.3f}")

URL=re.compile(r'https?://\S+|www\.\S+'); HTML=re.compile(r'<[^>]+>'); WS=re.compile(r'\s+')
def clean(s):
    s=unicodedata.normalize('NFKC',str(s)); s=HTML.sub(' ',s); s=URL.sub(' ',s)
    return WS.sub(' ',s).strip()
with Timer("clean"):
    for d in (train,test):
        d['T']=d['title'].apply(clean); d['C']=d['content'].apply(clean)

NUM=re.compile(r'\d+[.,]?\d*\s*%?'); YEAR=re.compile(r'\b(?:19|20)\d{2}\b')
NEG=['tidak','bukan','belum','tanpa','gagal','ditolak','membantah','bantah','sangkal','menyangkal','menolak']

with Timer("entity_mining"):
    up,lo=Counter(),Counter()
    for t in train['C']:
        for w in t.split():
            c=w.strip(_s.punctuation)
            if not c.isalpha() or len(c)<3: continue
            if c[:1].isupper(): up[c]+=1
            else: lo[c.capitalize()]+=1
    pur={w:up[w]/(up[w]+lo.get(w,0)) for w in up if up[w]>=CONFIG['entity_min_count']}
    tdf=Counter()
    for t in train['T']:
        for w in set(t.split()): tdf[w]+=1
    ENT={w for w,p in pur.items() if p>=CONFIG['entity_min_purity'] and tdf.get(w,0)/len(train)<CONFIG['entity_max_title_df']}
    print("entity pool:",len(ENT))
    for w in ['DKI','Jabar','Jokowi','Tidak','Corona']:
        if w in pur: print(f"   {w:8s} purity={pur[w]:.2f} in_pool={w in ENT}")

def chunks_of(s,w=CONFIG['chunk_words'],st=CONFIG['chunk_stride']):
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

def toks(s): return re.findall(r"[\w']+", s.lower())

with Timer("chunk_index"):
    allpairs=pd.concat([train[['ch','C']],test[['ch','C']]],ignore_index=True).drop_duplicates('ch')
    chunk_txt={};
    for h,c in zip(allpairs['ch'],allpairs['C']): chunk_txt[h]=chunks_of(c)
    df_cnt=Counter(); total_len=0; n_chunks=0
    chunk_tf={}
    for h,cs in chunk_txt.items():
        tfs=[]
        for c in cs:
            tk=toks(c); tf=Counter(tk); tfs.append((tf,len(tk)))
            total_len+=len(tk); n_chunks+=1
            for t in tf: df_cnt[t]+=1
        chunk_tf[h]=tfs
    AVGDL=total_len/max(n_chunks,1)
    IDF={t: math.log(1+(n_chunks-c+0.5)/(c+0.5)) for t,c in df_cnt.items()}
    print(f"articles {len(chunk_txt)}  chunks {n_chunks}  avg chunk len {AVGDL:.1f}  vocab {len(IDF)}")

k1,b=CONFIG['bm25_k1'],CONFIG['bm25_b']
def bm25_scores(title, ch):
    q=toks(title); out=[]
    for tf,L in chunk_tf[ch]:
        s=0.0
        for t in q:
            f=tf.get(t,0)
            if f: s+=IDF.get(t,0.0)*(f*(k1+1))/(f+k1*(1-b+b*L/AVGDL))
        out.append(s)
    return np.array(out) if out else np.array([0.0])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

with Timer("lsa"):
    corpus = pd.concat([train['T'],train['C'],test['T'],test['C']]).tolist()
    lsa_tfidf=TfidfVectorizer(ngram_range=(1,1),min_df=3,sublinear_tf=True,max_features=80000)
    Xc=lsa_tfidf.fit_transform(corpus)
    svd=TruncatedSVD(n_components=CONFIG['lsa_dim'],random_state=SEED)
    svd.fit(Xc)
    def embed(texts): return normalize(svd.transform(lsa_tfidf.transform(texts)))
    for d in (train,test):
        d_t=embed(d['T'].tolist()); d_c=embed(d['C'].tolist())
        d['lsa_cos']=np.sum(d_t*d_c,axis=1)
    print("explained variance:",round(float(svd.explained_variance_ratio_.sum()),4))
    print("lsa_cos by label:\n",train.groupby('label')['lsa_cos'].mean().round(4))

def build_features(df):
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
            _fp=[_first[e]/_n for e in _pres]
            _rf=[_tf[e]/max(_bmax,1) for e in _pres]
            sal_maxfirst=max(_fp); sal_meanfirst=sum(_fp)/len(_fp)
            sal_minrelfreq=min(_rf)
            sal_late=float(max(_fp)>0.5)
            sal_out_of_lead=float(max(_first[e] for e in _pres)>60)
            sal_singleton=float(min(_tf[e] for e in _pres)==1)
            sal_lead_frac=sum(1 for e in _pres if _first[e]<=60)/len(_pres)
        else:
            sal_maxfirst=sal_meanfirst=1.0; sal_minrelfreq=0.0
            sal_late=sal_out_of_lead=sal_singleton=1.0; sal_lead_frac=0.0
        sal_n_present=float(len(_pres))
        sal_all_present=float(bool(_tents) and len(_pres)==len(_tents))
        tset,cset=set(tw),set(cw)
        craw={w.strip(_s.punctuation) for w in C.split()}
        inter=tset&cset; union=tset|cset
        tb=set(zip(tw,tw[1:])); cb=set(zip(cw,cw[1:]))
        tents=[w for w in T.split() if w in ENT]
        miss=[w for w in tents if w not in craw]
        sig=0
        Ttok=T.split()
        for i,w in enumerate(Ttok):
            if w in ENT and w not in craw:
                nb=[Ttok[j] for j in (i-1,i+1) if 0<=j<len(Ttok)]
                nb=[x for x in nb if x.lower() not in ('dan','di','ke','yang')]
                if nb and all(x.strip(_s.punctuation) in craw for x in nb): sig+=1
        tn,cn=set(NUM.findall(T)),set(NUM.findall(C))
        ty,cy=set(YEAR.findall(T)),set(YEAR.findall(C))
        bm=bm25_scores(T,ch); order=np.argsort(-bm)
        topk=bm[order[:CONFIG['top_k']]]
        num_=sum(IDF.get(t,0.0) for t in tset&cset); den_=sum(IDF.get(t,0.0) for t in tset)
        rows.append(dict(
            n_shared=len(inter), jaccard=len(inter)/max(len(union),1),
            dice=2*len(inter)/max(len(tset)+len(cset),1),
            title_cov=len(inter)/max(len(tset),1), content_cov=len(inter)/max(len(cset),1),
            bigram_ov=len(tb&cb)/max(len(tb|cb),1),
            idf_title_cov=num_/max(den_,1e-6),
            n_title_ent=len(tents), n_title_ent_missing=len(miss),
            frac_title_ent_missing=len(miss)/max(len(tents),1),
            any_title_ent_missing=float(len(miss)>0), swap_signature=float(sig>0), n_swap_sig=sig,
            n_num_title=len(tn), n_num_missing=len(tn-cn),
            num_overlap=len(tn&cn)/max(len(tn),1) if tn else 1.0,
            num_mismatch=float(bool(tn) and not (tn&cn)),
            n_year_title=len(ty), year_missing=float(bool(ty) and not (ty&cy)),
            neg_title=float(any(w in tset for w in NEG)), neg_body=float(any(w in cset for w in NEG)),
            neg_mismatch=float(any(w in tset for w in NEG)!=any(w in cset for w in NEG)),
            bm25_max=float(bm.max()), bm25_mean=float(bm.mean()),
            bm25_top=float(topk.mean()), bm25_std=float(bm.std()),
            bm25_argmax_pos=float(order[0]/max(len(bm)-1,1)), n_chunks=len(bm),
            lsa_cos=float(lsa),
            len_title=len(tw), len_content=len(cw), len_ratio=len(tw)/max(len(cw),1),
            sal_maxfirst=sal_maxfirst, sal_meanfirst=sal_meanfirst,
            sal_minrelfreq=sal_minrelfreq, sal_late=sal_late,
            sal_out_of_lead=sal_out_of_lead, sal_singleton=sal_singleton,
            sal_lead_frac=sal_lead_frac, sal_n_present=sal_n_present,
            sal_all_present=sal_all_present,
        ))
    return pd.DataFrame(rows,index=df.index)

with Timer("features"):
    Ftr=build_features(train); Fte=build_features(test)
print(f"{Ftr.shape[1]} base features")

with Timer("structural_index"):
    chunk_owner=[]; flat_chunks=[]
    for h,cs in chunk_txt.items():
        for c in cs:
            chunk_owner.append(h); flat_chunks.append(c)
    print(f"articles {len(chunk_txt)}  total chunks {len(flat_chunks)}  avg/article {len(flat_chunks)/max(len(chunk_txt),1):.1f}")
    CHUNK_VEC = embed(flat_chunks)
    chunk_vecs=defaultdict(list)
    for h,v in zip(chunk_owner,CHUNK_VEC): chunk_vecs[h].append(v)
    for h in list(chunk_vecs.keys()): chunk_vecs[h]=np.array(chunk_vecs[h])

def structural_features(df):
    rows=[]
    tvecs=embed(df['T'].tolist())
    K=CONFIG['struct_topk']
    for tv,ch in zip(tvecs, df['ch']):
        svs=chunk_vecs.get(ch)
        if svs is None or len(svs)==0:
            rows.append(dict(struct_first_chunk=0.0, struct_first3_mean=0.0, struct_max_chunk=0.0,
                              struct_topk_mean=0.0, struct_std=0.0, struct_min_chunk=0.0,
                              struct_argmax_pos=0.0))
            continue
        sims = svs @ tv
        k=min(K,len(sims)); topk=np.sort(sims)[-k:]
        rows.append(dict(
            struct_first_chunk=float(sims[0]),
            struct_first3_mean=float(sims[:min(3,len(sims))].mean()),
            struct_max_chunk=float(sims.max()),
            struct_topk_mean=float(topk.mean()),
            struct_std=float(sims.std()),
            struct_min_chunk=float(sims.min()),
            struct_argmax_pos=float(np.argmax(sims)/max(len(sims)-1,1)),
        ))
    return pd.DataFrame(rows, index=df.index)

with Timer("structural_features"):
    Str_tr=structural_features(train); Str_te=structural_features(test)
    Ftr=pd.concat([Ftr,Str_tr],axis=1); Fte=pd.concat([Fte,Str_te],axis=1)
    FEATS=list(Ftr.columns)
print(f"{len(FEATS)} features total ({Str_tr.shape[1]} structural)")
print(pd.concat([Str_tr,train['label']],axis=1).groupby('label').mean().round(4).T)

def mine_targets(df_tr, min_count=2):
    si=Counter()
    for h,g in df_tr.groupby('ch'):
        pos=g[g.label==1]['T'].tolist(); neg=g[g.label==0]['T'].tolist()
        for p in pos:
            for n in neg:
                pt,nt=p.split(),n.split()
                op=[w for w in pt if w not in nt]; on=[w for w in nt if w not in pt]
                if len(op)==1 and len(on)==1: si[on[0]]+=1
    return {w for w,c in si.items() if c>=min_count}

def tamper_feats(df, TARG):
    a=[];b=[]
    for T,C in zip(df['T'],df['C']):
        craw={w.strip(_s.punctuation) for w in C.split()}
        ws=T.split()
        a.append(sum(1 for w in ws if w in TARG))
        b.append(float(any((w in TARG) and (w in ENT) and (w not in craw) for w in ws)))
    return np.column_stack([np.array(a,float), np.array(b,float)])

_T_all=mine_targets(train)
print(f"tamper-target vocabulary mined from full train: {len(_T_all)}")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import scipy.sparse as sp

def fit_sparse(tr_i, va_i, te=False):
    tv=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=20000,strip_accents='unicode')
    cv_=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=CONFIG['tfidf_word_max'],strip_accents='unicode')
    ct=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,sublinear_tf=True,max_features=CONFIG['tfidf_char_max'])
    A=tv.fit_transform(train['T'].iloc[tr_i]); B=cv_.fit_transform(train['C'].iloc[tr_i]); Cc=ct.fit_transform(train['T'].iloc[tr_i])
    Xtr=sp.hstack([A,B,Cc]).tocsr()
    def tf(df): return sp.hstack([tv.transform(df['T']),cv_.transform(df['C']),ct.transform(df['T'])]).tocsr()
    clf=LogisticRegression(max_iter=3000,class_weight='balanced',C=2.0,random_state=SEED)
    clf.fit(Xtr,y[tr_i])
    out_va=clf.predict_proba(tf(train.iloc[va_i]))[:,1]
    out_te=clf.predict_proba(tf(test))[:,1] if te else None
    return out_va,out_te

def fit_dense(tr_i,va_i,te=False):
    TARG=mine_targets(train.iloc[tr_i])          # fold-safe: mined from this fold's train rows only
    Xtr=np.column_stack([Ftr.iloc[tr_i].values, tamper_feats(train.iloc[tr_i],TARG)])
    Xva=np.column_stack([Ftr.iloc[va_i].values, tamper_feats(train.iloc[va_i],TARG)])
    if HAS_LGB:
        m=lgb.LGBMClassifier(n_estimators=600,learning_rate=0.05,num_leaves=31,
                              subsample=0.9,colsample_bytree=0.8,class_weight='balanced',
                              random_state=SEED,verbose=-1)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m=HistGradientBoostingClassifier(max_iter=500,learning_rate=0.05,random_state=SEED)
    m.fit(Xtr,y[tr_i])
    Xte=np.column_stack([Fte.values, tamper_feats(test,TARG)]) if te else None
    return m.predict_proba(Xva)[:,1], (m.predict_proba(Xte)[:,1] if te else None), m

with Timer("cv"):
    oof_s=np.zeros(len(train)); oof_d=np.zeros(len(train))
    for i,(a,b) in enumerate(folds):
        s,_=fit_sparse(a,b); d,_,m=fit_dense(a,b)
        oof_s[b]=s; oof_d[b]=d
        print(f"  fold{i} sparse={f1_score(y[b],(s>=0.5).astype(int),average='macro'):.4f} "
              f"dense={f1_score(y[b],(d>=0.5).astype(int),average='macro'):.4f}")
print("OOF sparse macroF1@0.5:",round(f1_score(y,(oof_s>=.5).astype(int),average='macro'),4))
print("OOF dense  macroF1@0.5:",round(f1_score(y,(oof_d>=.5).astype(int),average='macro'),4))
if HAS_LGB:
    imp=pd.Series(m.feature_importances_,index=FEATS+['n_tamper_target','tamper_target_missing']).sort_values(ascending=False)
    print("\ntop features:\n",imp.head(20))

with Timer("blend_threshold"):
    best=(None,-1)
    for wgt in np.arange(0,1.01,0.05):
        p=wgt*oof_s+(1-wgt)*oof_d
        sc=max(f1_score(y,(p>=t).astype(int),average='macro') for t in CONFIG['threshold_grid'])
        if sc>best[1]: best=(wgt,sc)
    W=best[0]; oof=W*oof_s+(1-W)*oof_d
    print(f"blend weight (sparse)={W:.2f}  cold-style OOF macroF1={best[1]:.4f}")

    _cnt=train['ch'].value_counts()
    cold_mask=train['ch'].map(_cnt).eq(1).values
    cold_oof=oof[cold_mask]; cold_y=y[cold_mask]
    n_cold=cold_mask.sum(); total=int(n_cold/CONFIG['test_mix']['A_cold'])
    print(f"cold-start analogue rows: {n_cold} (class0 {np.mean(cold_y==0):.3f}) target total {total}")

    N_SPLITS=20
    grid=CONFIG['threshold_grid']
    f1_curve=np.zeros(len(grid)); rules_only_curve=[]
    for split_seed in range(N_SPLITS):
        rs_=np.random.RandomState(split_seed); perm=rs_.permutation(len(train)); cut=int(0.85*len(train))
        tr_i,va_i=perm[:cut],perm[cut:]
        p_,b_=build_lookups(train.iloc[tr_i])
        reg_va,rule_va=rule_predict(train.iloc[va_i],p_,b_)
        y_va=train['label'].iloc[va_i].values; oof_va=oof[va_i]
        ys=[cold_y]; ws=[np.ones(len(cold_y))]
        base_preds=[np.ones(len(cold_y),int)]; base_ws=[np.ones(len(cold_y))]
        preds_at_t={t:[(cold_oof>=t).astype(int)] for t in grid}
        for r in ['C_exact','B_has_pos','B_no_pos']:
            m=reg_va==r; n=int(m.sum())
            if n==0: continue
            k=CONFIG['test_mix'][r]*total; w=np.full(n,k/n)
            yv=y_va[m]; rv=np.array([np.nan if l is None else l for l in np.array(rule_va,dtype=object)[m]],dtype=float)
            ov=oof_va[m]; defer=np.isnan(rv)
            ys.append(yv); ws.append(w)
            base_preds.append(np.where(defer,1,rv).astype(int)); base_ws.append(w)
            for t in grid:
                preds_at_t[t].append(np.where(defer,(ov>=t).astype(int),rv).astype(int))
        Y_ALL=np.concatenate(ys); W_ALL=np.concatenate(ws)
        for i,t in enumerate(grid):
            P_ALL=np.concatenate(preds_at_t[t])
            f1_curve[i]+=f1_score(Y_ALL,P_ALL,average='macro',sample_weight=W_ALL)
        base_P=np.concatenate(base_preds); base_W=np.concatenate(base_ws)
        rules_only_curve.append(f1_score(Y_ALL,base_P,average='macro',sample_weight=base_W))
    f1_curve/=N_SPLITS
    RULES_ONLY_MIX=float(np.mean(rules_only_curve))
    ti=int(np.argmax(f1_curve)); THR=float(grid[ti]); GLOBAL=float(f1_curve[ti])
    for i in range(0,len(grid),4):
        print(f"  t={grid[i]:.3f}  avg_global_f1={f1_curve[i]:.4f}")
print(f"chosen threshold {THR} -> {N_SPLITS}-split-averaged simulated global Macro F1 {GLOBAL:.4f}")
print(f"  rules-only under the same mixture: {RULES_ONLY_MIX:.4f} (std across splits {np.std(rules_only_curve):.4f})  -> model contributes {GLOBAL-RULES_ONLY_MIX:+.4f}")

with Timer("per_class_breakdown"):
    f1c_all=f1_score(y,(oof>=THR).astype(int),average=None)
    f1c_cold=f1_score(cold_y,(cold_oof>=THR).astype(int),average=None)
    print(f"model-ONLY (no rules), OOF @ threshold {THR}, ALL train rows (leak-free):")
    print(f"  class0 F1={f1c_all[0]:.4f}  class1 F1={f1c_all[1]:.4f}  macro={f1c_all.mean():.4f}")
    print(f"model-ONLY, OOF @ threshold {THR}, COLD-START subset only (n={cold_mask.sum()}, the model's real job):")
    print(f"  class0 F1={f1c_cold[0]:.4f}  class1 F1={f1c_cold[1]:.4f}  macro={f1c_cold.mean():.4f}")

with Timer("final"):
    idx=np.arange(len(train))
    s_va,s_te=fit_sparse(idx,idx[:1],te=True)
    d_va,d_te,mfull=fit_dense(idx,idx[:1],te=True)
    tp=W*s_te+(1-W)*d_te
    model_pred=(tp>=THR).astype(int)
    print("model positive rate on test:",round(model_pred.mean(),4))

with Timer("submit"):
    pair_all,byc_all=build_lookups(train)
    reg_t,rule_t=rule_predict(test,pair_all,byc_all)
    if CONFIG['USE_RULES']:
        final=np.array([model_pred[i] if rule_t[i] is None else rule_t[i] for i in range(len(test))])
    else:
        final=model_pred
    for r in ['A_cold','B_has_pos','B_no_pos','C_exact']:
        m=reg_t==r
        if m.sum(): print(f"  {r:10s} n={m.sum():5d}  positive rate={final[m].mean():.3f}")
    ov=sum(1 for i in range(len(test)) if rule_t[i] is not None and rule_t[i]!=model_pred[i])
    print("rows where a rule overrode the model:",ov)

    sub=pd.DataFrame({'id':test['id'],'label':final.astype(int)}).set_index('id').loc[test['id']].reset_index()
    assert sub.shape[0]==len(test) and set(sub['id'])==set(sample_sub['id'])
    assert set(sub['label'].unique())<={0,1} and list(sub.columns)==['id','label']
    sub.to_csv('/kaggle/working/submission.csv',index=False)
    sub_nr=pd.DataFrame({'id':test['id'],'label':model_pred.astype(int)})
    sub_nr.to_csv('/kaggle/working/submission_model_only.csv',index=False)
print(sub['label'].value_counts(normalize=True))

T=time.time()-RUN_T0
print("="*58)
print("V12 — V11 pipeline + structural multi-granularity matching")
print(f"OOF sparse/dense/blend : {f1_score(y,(oof_s>=.5).astype(int),average='macro'):.4f} / "
      f"{f1_score(y,(oof_d>=.5).astype(int),average='macro'):.4f} / {best[1]:.4f} (w={W:.2f})")
print(f"rules-only reference   : {RULES_ONLY:.4f}")
print(f"simulated global F1    : {GLOBAL:.4f} @ threshold {THR} (rules-only {RULES_ONLY_MIX:.4f}, model {GLOBAL-RULES_ONLY_MIX:+.4f})")
print(f"USE_RULES              : {CONFIG['USE_RULES']}")
print(f"total runtime          : {T/60:.1f} min  (CPU only)")
print("="*58)
for k,v in sorted(TIMINGS.items(),key=lambda kv:-kv[1]): print(f"  {k}: {v:.1f}s")