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
    USE_RULES=True,           # flip to False if the organizers disallow the lookup layer
    threshold_grid=[round(x,3) for x in np.arange(0.05,0.96,0.025)],
    test_mix={'A_cold':0.636,'C_exact':0.257,'B_has_pos':0.060,'B_no_pos':0.046},
)
print(json.dumps(CONFIG,indent=2,default=str))
with Timer("load"):
    F={'train.csv':r'D:/Lomba/IFEST2026_DAC/data/train.csv','test.csv':r'D:/Lomba/IFEST2026_DAC/data/test.csv','sample_submission.csv':r'D:/Lomba/IFEST2026_DAC/data/sample_submission.csv'}
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

def rule_predict(df,pair,byc):
    reg=[];lab=[]
    for th,ch in zip(df['th'],df['ch']):
        if (th,ch) in pair: reg.append('C_exact'); lab.append(pair[(th,ch)])
        elif ch in byc:
            if any(l==1 for _,l in byc[ch]): reg.append('B_has_pos'); lab.append(0)
            else: reg.append('B_no_pos'); lab.append(1)
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
    # global chunk statistics for BM25
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
IDF_TITLE = None
def build_features(df):
    rows=[]
    for T,C,ch,lsa in zip(df['T'],df['C'],df['ch'],df['lsa_cos']):
        tw=toks(T); cw=toks(C)
        tset,cset=set(tw),set(cw)
        craw={w.strip(_s.punctuation) for w in C.split()}
        inter=tset&cset; union=tset|cset
        tb=set(zip(tw,tw[1:])); cb=set(zip(cw,cw[1:]))
        tents=[w for w in T.split() if w in ENT]
        miss=[w for w in tents if w not in craw]
        # swap signature: entity missing while its title-neighbours ARE in the body
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
        # idf-weighted coverage: rare title terms are what get swapped
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
        ))
    return pd.DataFrame(rows,index=df.index)

with Timer("features"):
    Ftr=build_features(train); Fte=build_features(test)
    FEATS=list(Ftr.columns)
print(f"{len(FEATS)} features")
print(pd.concat([Ftr[['n_title_ent_missing','swap_signature','idf_title_cov','bm25_max','lsa_cos','num_mismatch']],
                 train['label']],axis=1).groupby('label').mean().round(4).T)
