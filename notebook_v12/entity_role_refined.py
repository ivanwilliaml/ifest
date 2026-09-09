"""Stage 2 of the revised A_cold plan: refined entity-role-consistency feature.
Stage 1 (chunk-distribution/'salient evidence') and Stage 3 (support/conflict score)
were already tested in ablation_experiment.py under different names and HURT performance
(0.6732->0.6668 and ->0.6694) -- not repeated here.

This builds a CLEANED entity pool (permissive purity, but excludes common institutional/
role words that were polluting the earlier 'expanded pool' detector: Gubernur, Presiden,
Menteri, Pemprov, COVID, Satgas, etc.) and adds ONE new feature: entity_role_conflict
(title entity missing from body AND an alternative entity from the clean pool is present
in the most-relevant body chunk). Added to the v12 48-feature baseline, re-evaluated via
the same grouped 5-fold CV.
"""
import os, sys, re, string as _s, unicodedata
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score
try:
    import lightgbm as lgb; HAS_LGB=True
except Exception:
    HAS_LGB=False
    from sklearn.ensemble import HistGradientBoostingClassifier

CACHE = r"D:\Lomba\IFEST2026_DAC\notebook_v12\_diag_cache"
Ftr = pd.read_csv(os.path.join(CACHE,'features.csv'))
train = pd.read_csv(os.path.join(CACHE,'train_slim.csv'))
y = train['label'].values
SEED=42

URL=re.compile(r'https?://\S+|www\.\S+'); HTML=re.compile(r'<[^>]+>'); WS=re.compile(r'\s+')
def clean(s):
    s=unicodedata.normalize('NFKC',str(s)); s=HTML.sub(' ',s); s=URL.sub(' ',s)
    return WS.sub(' ',s).strip()
train['T']=train['title'].apply(clean); train['C']=train['content'].apply(clean)
titles=train['T'].tolist(); bodies=train['C'].tolist()

# ---- cleaned permissive entity pool ----
STOPWORDS_ROLE = {
    'Pemprov','Pemkot','Pemkab','Gubernur','Wagub','Presiden','Wapres','Menteri','Menkes','Menko',
    'Kemenkes','Kementerian','Satgas','Ketua','Wali','Bupati','Wakil','Kepala','Dinas','Badan',
    'Komisi','Partai','Polri','TNI','DPR','DPRD','KPK','COVID','Corona','Covid','Kapolri','Kapolres',
    'Kapolda','Menkeu','Mendagri','Mensesneg','Kabareskrim','Kemendikbud','Kemdikbud','Kominfo',
    'Sekretaris','Direktur','Komisioner','Anggota','Pemerintah','Kemenag','BNPB','BPBD','Satpol',
    'Menristek','Panglima','Jaksa','Hakim','Rektor','Camat','Lurah','Kanwil','Kejaksaan','Pengadilan',
    'Kejagung','Kasatgas','Karo','Kabid','Kadin','Kadis',
}
up,lo=Counter(),Counter()
for t in train['C']:
    for w in t.split():
        c=w.strip(_s.punctuation)
        if not c.isalpha() or len(c)<3: continue
        if c[:1].isupper(): up[c]+=1
        else: lo[c.capitalize()]+=1
pur={w:up[w]/(up[w]+lo.get(w,0)) for w in up if up[w]>=5}
ENT_CLEAN = {w for w,p in pur.items() if p>=0.85 and w not in STOPWORDS_ROLE}
print(f"cleaned permissive entity pool: {len(ENT_CLEAN)} (excluded {len(STOPWORDS_ROLE)} role/institution stopwords)")

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

def top_chunk(title, chunks):
    tw=set(toks(title)); best_c, best_score = chunks[0], -1
    for c in chunks:
        score=len(tw & set(toks(c)))
        if score>best_score: best_score=score; best_c=c
    return best_c

def entity_role_conflict(title, body):
    Ttok=title.split()
    tents=[w for w in Ttok if w in ENT_CLEAN]
    craw={w.strip(_s.punctuation) for w in body.split()}
    missing=[w for w in tents if w not in craw]
    if not missing: return 0.0, 0.0  # (conflict_flag, n_missing_clean_ent)
    tc = top_chunk(title, chunks_of(body))
    chunk_ents=[w.strip(_s.punctuation) for w in tc.split() if w.strip(_s.punctuation) in ENT_CLEAN]
    for m in missing:
        if any(e!=m for e in chunk_ents):
            return 1.0, float(len(missing))
    return 0.0, float(len(missing))

print("computing entity_role_conflict feature for all rows...")
role_conf=[]; n_miss=[]
for T,C in zip(titles,bodies):
    f,n = entity_role_conflict(T,C)
    role_conf.append(f); n_miss.append(n)
Ftr2 = Ftr.copy()
Ftr2['entity_role_conflict'] = role_conf
Ftr2['n_clean_ent_missing'] = n_miss
print(f"entity_role_conflict fires on {sum(role_conf)}/{len(role_conf)} rows overall")

_cnt=train['content'].value_counts()
# use the SAME cold_mask logic as before via ch column proxy: recompute content hash
import hashlib
def nh(x):
    x=unicodedata.normalize('NFKC',str(x)).lower(); return re.sub(r'\s+',' ',x).strip()
train['ch']=train['content'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
_cnt2=train['ch'].value_counts()
cold_mask = train['ch'].map(_cnt2).eq(1).values
print(f"cold rows: {cold_mask.sum()}")

groups=train['ch'].values
sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
folds=list(sgkf.split(train,y,groups))

def run_cv(X):
    oof=np.zeros(len(train))
    for a,b in folds:
        Xa=X.iloc[a].values if hasattr(X,'iloc') else X[a]
        Xb=X.iloc[b].values if hasattr(X,'iloc') else X[b]
        if HAS_LGB:
            m=lgb.LGBMClassifier(n_estimators=600,learning_rate=0.05,num_leaves=31,subsample=0.9,
                                  colsample_bytree=0.8,class_weight='balanced',random_state=SEED,verbose=-1)
        else:
            m=HistGradientBoostingClassifier(max_iter=500,learning_rate=0.05,random_state=SEED)
        m.fit(Xa,y[a])
        oof[b]=m.predict_proba(Xb)[:,1]
    return oof

def report(oof, name):
    cold_idx=np.where(cold_mask)[0]
    co=oof[cold_idx]; cy=y[cold_idx]
    best=(0.5,-1)
    for t in np.arange(0.05,0.96,0.025):
        mac=f1_score(cy,(co>=t).astype(int),average='macro')
        if mac>best[1]: best=(t,mac)
    pred=(co>=best[0]).astype(int)
    f1c=f1_score(cy,pred,average=None,labels=[0,1])
    p0=precision_score(cy,pred,pos_label=0,zero_division=0); r0=recall_score(cy,pred,pos_label=0,zero_division=0)
    print(f"{name:45s} macroF1={best[1]:.4f} F1_0={f1c[0]:.4f} F1_1={f1c[1]:.4f} P0={p0:.4f} R0={r0:.4f}")
    return best[1], f1c[0], f1c[1]

print("\n=== baseline v12 (48 features) ===")
oof_base = run_cv(Ftr)
report(oof_base, "baseline (48 feats)")

print("\n=== + entity_role_conflict (refined, cleaned pool) ===")
oof_new = run_cv(Ftr2)
report(oof_new, "+ entity_role_conflict (50 feats)")

print("\nreference: original numbers from earlier diagnostic run: macroF1=0.6509 F1_0=0.3403 F1_1=0.9615")
