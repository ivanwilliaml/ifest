"""Compliant next steps for A_cold: (1) expanded BM25 chunk-retrieval features added to
the winning CatBoost/XGB/RF ensemble, (2) class-weight sweep on the ensemble. No
pretrained embeddings/NLI (explicitly banned by competition rules -- XLM-R named
directly). Body has no punctuation (verified earlier) so "sentences" = 50-word chunks,
same infra as before.
"""
import os, sys, time, hashlib, re, unicodedata, math, string as _s
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

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

# ---- from-scratch BM25 over 50-word chunks (compliant, explicitly permitted) ----
def toks(s): return re.findall(r"[\w']+", s.lower())
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
log("building BM25 corpus stats...")
chunk_lists = [chunks_of(b) for b in bodies]
df_cnt=Counter(); total_len=0; n_chunks=0; chunk_tf_all=[]
for cs in chunk_lists:
    tfs=[]
    for c in cs:
        tk=toks(c); tf=Counter(tk); tfs.append((tf,len(tk)))
        total_len+=len(tk); n_chunks+=1
        for t in tf: df_cnt[t]+=1
    chunk_tf_all.append(tfs)
AVGDL=total_len/max(n_chunks,1)
IDF={t: math.log(1+(n_chunks-c+0.5)/(c+0.5)) for t,c in df_cnt.items()}
k1,b_=1.5,0.75
def bm25_row(title, tfs):
    q=toks(title); out=[]
    for tf,L in tfs:
        s=0.0
        for t in q:
            f=tf.get(t,0)
            if f: s+=IDF.get(t,0.0)*(f*(k1+1))/(f+k1*(1-b_+b_*L/AVGDL))
        out.append(s)
    return np.array(out) if out else np.array([0.0])

log("computing expanded BM25 chunk-retrieval features (max/top3/top5/std/first-rank/position)...")
rows=[]
for T, tfs in zip(titles, chunk_tf_all):
    bm=bm25_row(T,tfs)
    order=np.argsort(-bm); n=len(bm)
    top3=bm[order[:min(3,n)]].mean(); top5=bm[order[:min(5,n)]].mean()
    first_rank = float(order[0])/max(n-1,1)  # position of the single best-matching chunk
    rows.append(dict(
        bm25s_max=float(bm.max()), bm25s_top3_mean=float(top3), bm25s_top5_mean=float(top5),
        bm25s_std=float(bm.std()), bm25s_first_rank=first_rank,
        bm25s_score_at_first_chunk=float(bm[0]),  # relevance of the literal first chunk (lead paragraph)
    ))
BMdf = pd.DataFrame(rows)
log(f"BM25-sentence features: {BMdf.shape[1]}")

Ftr_v2 = pd.concat([Ftr[FEAT48], BMdf], axis=1)
FEAT_V2 = list(Ftr_v2.columns)

def run_cv_model(model_fn, Xdf):
    oof=np.zeros(len(train))
    Xv = Xdf.values
    for a,b in folds:
        m = model_fn()
        m.fit(Xv[a],y[a]); oof[b]=m.predict_proba(Xv[b])[:,1]
    return oof

def best_macro(p, yy=None):
    yy = cy if yy is None else yy
    best=(0.5,-1)
    for t in np.arange(0.05,0.96,0.025):
        mac=f1_score(yy,(p>=t).astype(int),average='macro')
        if mac>best[1]: best=(t,mac)
    return best

def report(oof, name):
    co=oof[cold_idx]
    t,mac=best_macro(co)
    pred=(co>=t).astype(int)
    f1c=f1_score(cy,pred,average=None,labels=[0,1])
    p0=precision_score(cy,pred,pos_label=0,zero_division=0); r0=recall_score(cy,pred,pos_label=0,zero_division=0)
    print(f"{name:45s} macroF1={mac:.4f} F1_0={f1c[0]:.4f} F1_1={f1c[1]:.4f} P0={p0:.4f} R0={r0:.4f}")
    return mac

log("=== Exp 1: baseline 48-feature ensemble (reference) vs + BM25-sentence features ===")
def make_cat(): return cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                                              auto_class_weights='Balanced', random_seed=SEED, verbose=False)
def make_xgb():
    return xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=0.03, subsample=0.9,
                              colsample_bytree=0.8, scale_pos_weight=17.0, random_state=SEED,
                              eval_metric='logloss')
def make_rf(): return RandomForestClassifier(n_estimators=800, max_depth=12, class_weight='balanced',
                                              random_state=SEED, n_jobs=-1)

oof_cat48 = run_cv_model(make_cat, Ftr[FEAT48]); report(oof_cat48,"48-feat CatBoost (reference)")
oof_cat_v2 = run_cv_model(make_cat, Ftr_v2); report(oof_cat_v2,"+BM25-sentence CatBoost")

oof_xgb48 = run_cv_model(make_xgb, Ftr[FEAT48]); report(oof_xgb48,"48-feat XGBoost (reference)")
oof_xgb_v2 = run_cv_model(make_xgb, Ftr_v2); report(oof_xgb_v2,"+BM25-sentence XGBoost")

oof_rf48 = run_cv_model(make_rf, Ftr[FEAT48]); report(oof_rf48,"48-feat RandomForest (reference)")
oof_rf_v2 = run_cv_model(make_rf, Ftr_v2); report(oof_rf_v2,"+BM25-sentence RandomForest")

log("=== Exp 2: re-search ensemble weights with expanded features ===")
best=(None,-1)
for w2 in np.arange(0,1.01,0.1):
    for w3 in np.arange(0,1.01-w2,0.1):
        w4=1-w2-w3
        if w4<0: continue
        blend = w2*oof_cat_v2 + w3*oof_xgb_v2 + w4*oof_rf_v2
        _,mac = best_macro(blend[cold_idx])
        if mac>best[1]: best=((w2,w3,w4),mac)
print(f"best weights cat/xgb/rf (with BM25-sentence feats) = {best[0]}  macroF1={best[1]:.4f}")
w2,w3,w4=best[0]
oof_best_v2 = w2*oof_cat_v2+w3*oof_xgb_v2+w4*oof_rf_v2
report(oof_best_v2, "BEST BLEND (48+BM25-sent)")
print(f"reference (old best, 48 feats only): macroF1=0.6571 F1_0=0.3501")

log("=== Exp 3: class-weight sweep on CatBoost (48+BM25-sent feats) ===")
best_cw=(None,-1)
for w0 in [2,3,4,6,8]:
    def make_cat_w(w0=w0):
        return cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                                      class_weights=[w0,1], random_seed=SEED, verbose=False)
    oof_w = run_cv_model(make_cat_w, Ftr_v2)
    mac = report(oof_w, f"CatBoost class_weight 0:{w0}:1")
    if mac>best_cw[1]: best_cw=(w0,mac,oof_w)

print(f"\nbest class weight for CatBoost: 0:{best_cw[0]}:1  macroF1={best_cw[1]:.4f}")

np.save(os.path.join(CACHE,'oof_cat_v2.npy'), oof_cat_v2)
np.save(os.path.join(CACHE,'oof_xgb_v2.npy'), oof_xgb_v2)
np.save(os.path.join(CACHE,'oof_rf_v2.npy'), oof_rf_v2)
np.save(os.path.join(CACHE,'oof_best_v2.npy'), oof_best_v2)
log(f"total runtime: {(time.time()-T0)/60:.1f} min")
