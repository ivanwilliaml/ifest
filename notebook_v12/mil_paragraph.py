"""Paragraph-level MIL for A_cold. Data has NO punctuation/newlines (verified: 0 in raw
content) so "paragraph" = non-overlapping ~80-word chunk (best available proxy). Bag =
article (label = article label), instances = (title, paragraph) pairs. Train a
paragraph-level CatBoost with article-label-as-instance-label (standard MIL simplification),
then aggregate paragraph predictions back to article level via several pooling rules.
"""
import os, sys, time, hashlib, re, unicodedata, math, string as _s
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, precision_score, recall_score
import catboost as cb

SEED=42
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

CACHE = r"D:\Lomba\IFEST2026_DAC\notebook_v12\_diag_cache"
Ftr = pd.read_csv(os.path.join(CACHE,'features.csv'))
train = pd.read_csv(os.path.join(CACHE,'train_slim.csv'))
y = train['label'].values
FEAT48 = [c for c in Ftr.columns if c != 'action_conflict']

URL=re.compile(r'https?://\S+|www\.\S+'); HTML=re.compile(r'<[^>]+>'); WS=re.compile(r'\s+')
def clean(s):
    s=unicodedata.normalize('NFKC',str(s)); s=HTML.sub(' ',s); s=URL.sub(' ',s)
    return WS.sub(' ',s).strip()
train['T']=train['title'].apply(clean); train['C']=train['content'].apply(clean)
titles=train['T'].tolist(); bodies=train['C'].tolist()
def nh(x):
    x=unicodedata.normalize('NFKC',str(x)).lower(); return re.sub(r'\s+',' ',x).strip()
train['ch']=train['content'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
_cnt=train['ch'].value_counts()
cold_mask = train['ch'].map(_cnt).eq(1).values
cold_idx = np.where(cold_mask)[0]; cy = y[cold_idx]
log(f"cold rows: {cold_mask.sum()}")

groups=train['ch'].values
sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
folds=list(sgkf.split(train,y,groups))

# ---- entity pool + NUM/NEG (same as v12) ----
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
NUM=re.compile(r'\d+[.,]?\d*\s*%?')
NEG=['tidak','bukan','belum','tanpa','gagal','ditolak','membantah','bantah','sangkal','menyangkal','menolak']
def toks(s): return re.findall(r"[\w']+", s.lower())

# ---- paragraph proxy: non-overlapping 80-word chunks ----
PARA_W=80
def paragraphs_of(s):
    ws=s.split()
    if not ws: return ['']
    out=[' '.join(ws[i:i+PARA_W]) for i in range(0,len(ws),PARA_W)]
    return out if out else ['']

log("splitting bodies into paragraph-proxy chunks...")
para_lists = [paragraphs_of(b) for b in bodies]
n_paras = [len(p) for p in para_lists]
log(f"avg paragraphs/article: {np.mean(n_paras):.2f}  (min={min(n_paras)} max={max(n_paras)})")

# ---- BM25 over paragraphs (for paragraph-level bm25 feature + retrieval-based feats) ----
log("building BM25 over paragraph units...")
df_cnt=Counter(); total_len=0; n_chunks=0; para_tf=[]
for ps in para_lists:
    tfs=[]
    for p in ps:
        tk=toks(p); tf=Counter(tk); tfs.append((tf,len(tk)))
        total_len+=len(tk); n_chunks+=1
        for t in tf: df_cnt[t]+=1
    para_tf.append(tfs)
AVGDL=total_len/max(n_chunks,1)
IDF={t: math.log(1+(n_chunks-c+0.5)/(c+0.5)) for t,c in df_cnt.items()}
k1,b_=1.5,0.75
def bm25_para(title, tfs):
    q=toks(title); out=[]
    for tf,L in tfs:
        s=0.0
        for t in q:
            f=tf.get(t,0)
            if f: s+=IDF.get(t,0.0)*(f*(k1+1))/(f+k1*(1-b_+b_*L/AVGDL))
        out.append(s)
    return out

# ---- LSA embed (from-scratch) for title/paragraph cosine ----
log("fitting from-scratch LSA (title + all paragraphs)...")
all_paras_flat = [p for ps in para_lists for p in ps]
corpus = titles + all_paras_flat
lsa_tfidf=TfidfVectorizer(ngram_range=(1,1),min_df=3,sublinear_tf=True,max_features=80000)
Xc=lsa_tfidf.fit_transform(corpus)
svd=TruncatedSVD(n_components=160,random_state=SEED); svd.fit(Xc)
def embed(texts): return normalize(svd.transform(lsa_tfidf.transform(texts)))
title_vecs = embed(titles)
log("embedding paragraphs...")
para_vecs_flat = embed(all_paras_flat)
# re-split back per article
para_vecs=[]; off=0
for ps in para_lists:
    para_vecs.append(para_vecs_flat[off:off+len(ps)]); off+=len(ps)

log("building paragraph-instance feature rows...")
inst_rows=[]; inst_article_idx=[]; inst_label=[]
for ai in range(len(train)):
    T=titles[ai]; tw=toks(T); tset=set(tw)
    tents=[w for w in T.split() if w in ENT]
    tn=set(NUM.findall(T)); neg_t=any(w in tset for w in NEG)
    tv=title_vecs[ai]
    bm=bm25_para(T, para_tf[ai])
    for pi, (p, pv, bscore) in enumerate(zip(para_lists[ai], para_vecs[ai], bm)):
        cw=toks(p); cset=set(cw); craw={w.strip(_s.punctuation) for w in p.split()}
        inter=tset&cset; union=tset|cset
        miss=[w for w in tents if w not in craw]
        cn=set(NUM.findall(p)); neg_c=any(w in cset for w in NEG)
        cos = float(np.dot(tv,pv))
        inst_rows.append(dict(
            jaccard=len(inter)/max(len(union),1), title_cov=len(inter)/max(len(tset),1),
            n_title_ent=len(tents), n_ent_missing=len(miss),
            frac_ent_missing=len(miss)/max(len(tents),1), any_ent_missing=float(len(miss)>0),
            num_mismatch=float(bool(tn) and not (tn&cn)),
            neg_mismatch=float(neg_t!=neg_c), bm25=float(bscore), lsa_cos=cos,
            para_position=pi/max(len(para_lists[ai])-1,1), para_len=len(cw),
        ))
        inst_article_idx.append(ai); inst_label.append(y[ai])
InstDf = pd.DataFrame(inst_rows)
inst_article_idx=np.array(inst_article_idx); inst_label=np.array(inst_label)
log(f"total paragraph instances: {len(InstDf)}")

INST_FEATS = list(InstDf.columns)
Xinst = InstDf.values

log("=== training paragraph-level CatBoost (article label as instance label, MIL) ===")
inst_oof = np.zeros(len(InstDf))
for a,b in folds:  # a,b are ARTICLE indices
    inst_a = np.isin(inst_article_idx, a); inst_b = np.isin(inst_article_idx, b)
    m = cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                               class_weights=[4,1], random_seed=SEED, verbose=False)
    m.fit(Xinst[inst_a], inst_label[inst_a])
    inst_oof[inst_b] = m.predict_proba(Xinst[inst_b])[:,1]
    log(f"  fold done ({inst_a.sum()} train instances, {inst_b.sum()} val instances)")

log("aggregating paragraph OOF predictions back to article level...")
article_para_scores = defaultdict(list)
for idx, ai in enumerate(inst_article_idx):
    article_para_scores[ai].append(inst_oof[idx])

def aggregate(scores, method):
    s = np.sort(scores)[::-1]
    if method=='max': return s[0]
    if method=='top2': return s[:2].mean()
    if method=='top3': return s[:3].mean()
    if method=='top5': return s[:5].mean()
    if method=='p90': return np.percentile(scores,90)

def best_macro(p, yy):
    best=(0.5,-1)
    for t in np.arange(0.05,0.96,0.025):
        mac=f1_score(yy,(p>=t).astype(int),average='macro')
        if mac>best[1]: best=(t,mac)
    return best

def report(oof_article, name):
    co=oof_article[cold_idx]
    t,mac=best_macro(co,cy)
    pred=(co>=t).astype(int); f1c=f1_score(cy,pred,average=None,labels=[0,1])
    p0=precision_score(cy,pred,pos_label=0,zero_division=0); r0=recall_score(cy,pred,pos_label=0,zero_division=0)
    print(f"{name:30s} macroF1={mac:.4f} F1_0={f1c[0]:.4f} F1_1={f1c[1]:.4f} P0={p0:.4f} R0={r0:.4f}")
    return mac

log("=== computing whole-body baseline (48 feats, CatBoost w=4:1, for the 0.7/0.3 blend) ===")
def make_cat(): return cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                                              class_weights=[4,1], random_seed=SEED, verbose=False)
oof_whole = np.zeros(len(train))
Xw = Ftr[FEAT48].values
for a,b in folds:
    m=make_cat(); m.fit(Xw[a],y[a]); oof_whole[b]=m.predict_proba(Xw[b])[:,1]
report(oof_whole, "whole-body baseline (48 feats)")

results={}
for method in ['max','top2','top3','top5','p90']:
    oof_agg = np.zeros(len(train))
    for ai, scores in article_para_scores.items():
        oof_agg[ai] = aggregate(scores, method)
    mac = report(oof_agg, f"paragraph {method}")
    results[method]=oof_agg

log("=== 0.7*best_paragraph(max) + 0.3*whole_body ===")
oof_blend = 0.7*results['max'] + 0.3*oof_whole
report(oof_blend, "0.7*para_max + 0.3*whole")

log("=== 0.7*best_paragraph(top3) + 0.3*whole_body ===")
oof_blend2 = 0.7*results['top3'] + 0.3*oof_whole
report(oof_blend2, "0.7*para_top3 + 0.3*whole")

print(f"\nreference baseline (whole-body 48-feat + BM25-sent, from prior run): macroF1=0.6586 F1_0=0.3540")
log(f"total runtime: {(time.time()-T0)/60:.1f} min")
