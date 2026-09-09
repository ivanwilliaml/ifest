"""B_no_pos ONLY: headline-vs-OWN-BODY contradiction features (not title-vs-negative-title
retrieval). Question: do the ~34 negatives carry intrinsic headline-body signal we haven't
exploited yet? Simple model (Logistic Regression / single combined score), honest repeated
CV with fit/threshold/eval strictly separated.
"""
import re, time, hashlib, unicodedata, string as _s
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

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
train['T']=train['title'].apply(clean); train['C']=train['content'].apply(clean)
titles=train['T'].tolist(); bodies=train['C'].tolist(); labels=train['label'].values

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

by_body=defaultdict(list)
for i,ch in enumerate(train['ch']): by_body[ch].append(i)

def body_contradiction(title, body):
    tw = title.split(); tset=set(w.lower() for w in tw)
    craw = {w.strip(_s.punctuation) for w in body.split()}
    cset = set(w.lower() for w in body.split())
    tents = [w for w in tw if w in ENT]; miss = [w for w in tents if w not in craw]
    tn, cn = set(NUM.findall(title)), set(NUM.findall(body))
    ty, cy = set(YEAR.findall(title)), set(YEAR.findall(body))
    neg_t = any(w in tset for w in NEG); neg_c = any(w in cset for w in NEG)
    inter = tset & cset; union = tset | cset
    action_hit = 0.0
    for w1,w2 in ANTONYMS:
        if (w1 in tset and w2 in cset) or (w2 in tset and w1 in cset): action_hit = 1.0; break
    return dict(
        jaccard=len(inter)/max(len(union),1), title_cov=len(inter)/max(len(tset),1),
        ent_missing=float(len(miss)>0), frac_ent_missing=len(miss)/max(len(tents),1),
        n_title_ent=float(len(tents)),
        num_mismatch=float(bool(tn) and not (tn&cn)), year_mismatch=float(bool(ty) and not (ty&cy)),
        neg_mismatch=float(neg_t != neg_c), action_conflict=action_hit,
    )

log("collecting B_no_pos pool (leave-one-title-out)...")
pool=[]
for ch, idxs in by_body.items():
    if len(idxs)<2: continue
    if len(set(train['th'].iloc[idxs]))<2: continue
    for i in idxs:
        memory=[j for j in idxs if j!=i]
        if any(labels[j]==1 for j in memory): continue
        if not any(labels[j]==0 for j in memory): continue
        pool.append(i)
log(f"pool size: {len(pool)}")

rows=[]
for i in pool:
    f = body_contradiction(titles[i], bodies[i])
    f['label']=labels[i]
    rows.append(f)
FB=pd.DataFrame(rows)
y=FB['label'].values
feat_cols=[c for c in FB.columns if c!='label']
X=FB[feat_cols].values
log(f"feature matrix: {X.shape} class0={int((y==0).sum())} class1={int((y==1).sum())}")
print(FB.groupby('label').mean().round(4).T)

def honest_repeated_cv(model_fn, X, y, n_repeats=5, n_splits=5):
    records=[]
    for rep in range(n_repeats):
        kf=KFold(n_splits=n_splits, shuffle=True, random_state=SEED+rep)
        for tr_i, va_i in kf.split(X):
            rs=np.random.RandomState(SEED+rep)
            perm=rs.permutation(len(tr_i)); half=len(tr_i)//2
            fit_i, thr_i = tr_i[perm[:half]], tr_i[perm[half:]]
            m = model_fn()
            m.fit(X[fit_i], y[fit_i])
            p_thr = m.predict_proba(X[thr_i])[:,1]
            best=(0.5,-1)
            for t in np.arange(0.05,0.96,0.025):
                mac=f1_score(y[thr_i],(p_thr>=t).astype(int),average='macro')
                if mac>best[1]: best=(t,mac)
            thr=best[0]
            p_va=m.predict_proba(X[va_i])[:,1]
            pred=(p_va>=thr).astype(int)
            f1c=f1_score(y[va_i],pred,average=None,labels=[0,1])
            acc=accuracy_score(y[va_i],pred)
            records.append(dict(rep=rep,threshold=thr,acc=acc,macro_f1=f1c.mean(),f1_0=f1c[0],f1_1=f1c[1]))
    return pd.DataFrame(records)

log("=== C1: Logistic Regression, all 9 body-contradiction features ===")
r1 = honest_repeated_cv(lambda: LogisticRegression(class_weight='balanced', max_iter=1000), X, y)
print(f"macroF1={r1.macro_f1.mean():.4f}(+/-{r1.macro_f1.std():.4f}) F1_0={r1.f1_0.mean():.4f}(+/-{r1.f1_0.std():.4f}) F1_1={r1.f1_1.mean():.4f}")

log("=== C2: single combined score (sum of conflict indicators) + threshold ===")
conflict_score = (FB['frac_ent_missing'] + FB['ent_missing'] + FB['num_mismatch']
                   + FB['year_mismatch'] + FB['neg_mismatch'] + FB['action_conflict']
                   - FB['jaccard'] - FB['title_cov']).values.reshape(-1,1)
class ScoreModel:
    def fit(self, X, y): return self
    def predict_proba(self, X):
        s = X.ravel(); s = (s - s.min())/(s.max()-s.min()+1e-9)
        return np.column_stack([1-s, s])
r2 = honest_repeated_cv(lambda: ScoreModel(), conflict_score, y)
print(f"macroF1={r2.macro_f1.mean():.4f}(+/-{r2.macro_f1.std():.4f}) F1_0={r2.f1_0.mean():.4f}(+/-{r2.f1_0.std():.4f}) F1_1={r2.f1_1.mean():.4f}")

log("=== C3: single feature -- entity/frac_ent_missing alone ===")
r3 = honest_repeated_cv(lambda: LogisticRegression(class_weight='balanced', max_iter=1000),
                          FB[['frac_ent_missing']].values, y)
print(f"macroF1={r3.macro_f1.mean():.4f}(+/-{r3.macro_f1.std():.4f}) F1_0={r3.f1_0.mean():.4f}(+/-{r3.f1_0.std():.4f}) F1_1={r3.f1_1.mean():.4f}")

log("=== C4: LogReg, only the 3 strongest-looking features (by class mean gap) ===")
gaps = (FB.groupby('label').mean().diff().iloc[-1].abs().sort_values(ascending=False))
top3 = [c for c in gaps.index if c in feat_cols][:3]
print(f"  top3 by class-mean gap: {top3}")
r4 = honest_repeated_cv(lambda: LogisticRegression(class_weight='balanced', max_iter=1000),
                          FB[top3].values, y)
print(f"macroF1={r4.macro_f1.mean():.4f}(+/-{r4.macro_f1.std():.4f}) F1_0={r4.f1_0.mean():.4f}(+/-{r4.f1_0.std():.4f}) F1_1={r4.f1_1.mean():.4f}")

log("=== SUMMARY ===")
summ = pd.DataFrame([
    dict(method='baseline: hard rule (always 1)', macro_f1=0.4881, f1_0=0.0, f1_1=0.9763),
    dict(method='retrieval-only honest CV (max_char)', macro_f1=0.5755, f1_0=0.2253, f1_1=0.9257),
    dict(method='C1: LogReg all 9 body-contradiction feats', macro_f1=r1.macro_f1.mean(), f1_0=r1.f1_0.mean(), f1_1=r1.f1_1.mean()),
    dict(method='C2: single combined conflict score', macro_f1=r2.macro_f1.mean(), f1_0=r2.f1_0.mean(), f1_1=r2.f1_1.mean()),
    dict(method='C3: entity-missing alone', macro_f1=r3.macro_f1.mean(), f1_0=r3.f1_0.mean(), f1_1=r3.f1_1.mean()),
    dict(method=f'C4: top3 features ({top3})', macro_f1=r4.macro_f1.mean(), f1_0=r4.f1_0.mean(), f1_1=r4.f1_1.mean()),
])
print(summ.round(4).to_string(index=False))
summ.to_csv('bnopos_contradiction_summary.csv', index=False)
log(f"total runtime: {(time.time()-T0)/60:.1f} min")
