"""Multi-model stacking for A_cold: LightGBM, CatBoost, XGBoost, RandomForest (all on the
same 48-feature v12 baseline) + the already-computed V14 neural OOF. Blend/stack via OOF
to see if a combination beats the single-best LightGBM (macroF1=0.6422).
"""
import os, sys, time, hashlib, re, unicodedata
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

SEED=42
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

CACHE = r"D:\Lomba\IFEST2026_DAC\notebook_v12\_diag_cache"
Ftr = pd.read_csv(os.path.join(CACHE,'features.csv'))
train = pd.read_csv(os.path.join(CACHE,'train_slim.csv'))
y = train['label'].values
FEAT_COLS = [c for c in Ftr.columns if c != 'action_conflict']
X = Ftr[FEAT_COLS].values

def nh(x):
    x=unicodedata.normalize('NFKC',str(x)).lower(); return re.sub(r'\s+',' ',x).strip()
train['ch']=train['content'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
_cnt=train['ch'].value_counts()
cold_mask = train['ch'].map(_cnt).eq(1).values
cold_idx = np.where(cold_mask)[0]
cy = y[cold_idx]
log(f"cold rows: {cold_mask.sum()}")

groups=train['ch'].values
sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
folds=list(sgkf.split(train,y,groups))

def best_macro(p, yy=None):
    yy = cy if yy is None else yy
    best=(0.5,-1)
    for t in np.arange(0.05,0.96,0.025):
        mac=f1_score(yy,(p>=t).astype(int),average='macro')
        if mac>best[1]: best=(t,mac)
    return best

def report(oof, name):
    co=oof[cold_idx]
    t,mac = best_macro(co)
    pred=(co>=t).astype(int)
    f1c=f1_score(cy,pred,average=None,labels=[0,1])
    p0=precision_score(cy,pred,pos_label=0,zero_division=0); r0=recall_score(cy,pred,pos_label=0,zero_division=0)
    print(f"{name:30s} macroF1={mac:.4f} F1_0={f1c[0]:.4f} F1_1={f1c[1]:.4f} P0={p0:.4f} R0={r0:.4f}")
    return mac

log("=== LightGBM (current best config) ===")
oof_lgb=np.zeros(len(train))
for a,b in folds:
    m=lgb.LGBMClassifier(n_estimators=600,num_leaves=31,learning_rate=0.05,subsample=0.9,
                          colsample_bytree=0.8,class_weight='balanced',random_state=SEED,verbose=-1)
    m.fit(X[a],y[a]); oof_lgb[b]=m.predict_proba(X[b])[:,1]
report(oof_lgb,"LightGBM")

log("=== CatBoost ===")
oof_cat=np.zeros(len(train))
for a,b in folds:
    m=cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03, auto_class_weights='Balanced',
                             random_seed=SEED, verbose=False)
    m.fit(X[a],y[a]); oof_cat[b]=m.predict_proba(X[b])[:,1]
report(oof_cat,"CatBoost")

log("=== XGBoost ===")
oof_xgb=np.zeros(len(train))
for a,b in folds:
    n0=(y[a]==0).sum(); n1=(y[a]==1).sum()
    m=xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=0.03, subsample=0.9,
                         colsample_bytree=0.8, scale_pos_weight=n1/n0, random_state=SEED,
                         eval_metric='logloss', use_label_encoder=False)
    m.fit(X[a],y[a]); oof_xgb[b]=m.predict_proba(X[b])[:,1]
report(oof_xgb,"XGBoost")

log("=== RandomForest ===")
oof_rf=np.zeros(len(train))
for a,b in folds:
    m=RandomForestClassifier(n_estimators=800, max_depth=12, class_weight='balanced',
                              random_state=SEED, n_jobs=-1)
    m.fit(X[a],y[a]); oof_rf[b]=m.predict_proba(X[b])[:,1]
report(oof_rf,"RandomForest")

log("loading V14 neural OOF...")
oof_v14 = np.load(os.path.join(CACHE,'oof_v14.npy'))
report(oof_v14,"V14 neural")

log("=== simple average blend (all 5) ===")
oof_avg = (oof_lgb+oof_cat+oof_xgb+oof_rf+oof_v14)/5
report(oof_avg,"avg(lgb,cat,xgb,rf,v14)")

log("=== simple average blend (trees only, no neural) ===")
oof_avg_trees = (oof_lgb+oof_cat+oof_xgb+oof_rf)/4
report(oof_avg_trees,"avg(lgb,cat,xgb,rf)")

log("=== grid weight search (trees only) ===")
best=(None,-1)
names=['lgb','cat','xgb','rf']
oofs=[oof_lgb,oof_cat,oof_xgb,oof_rf]
for w1 in np.arange(0,1.01,0.25):
    for w2 in np.arange(0,1.01-w1,0.25):
        for w3 in np.arange(0,1.01-w1-w2,0.25):
            w4=1-w1-w2-w3
            if w4<0: continue
            blend = w1*oof_lgb+w2*oof_cat+w3*oof_xgb+w4*oof_rf
            _,mac = best_macro(blend[cold_idx])
            if mac>best[1]: best=((w1,w2,w3,w4),mac)
print(f"best weights lgb/cat/xgb/rf = {best[0]}  macroF1={best[1]:.4f}")
w1,w2,w3,w4 = best[0]
oof_best_blend = w1*oof_lgb+w2*oof_cat+w3*oof_xgb+w4*oof_rf
report(oof_best_blend, "BEST WEIGHTED BLEND")
np.save(os.path.join(CACHE,'oof_lgb.npy'), oof_lgb)
np.save(os.path.join(CACHE,'oof_cat.npy'), oof_cat)
np.save(os.path.join(CACHE,'oof_xgb.npy'), oof_xgb)
np.save(os.path.join(CACHE,'oof_rf.npy'), oof_rf)
np.save(os.path.join(CACHE,'oof_best_blend.npy'), oof_best_blend)

log("=== stacking: logistic regression meta-learner on OOF probs (5-fold nested) ===")
meta_X = np.column_stack([oof_lgb,oof_cat,oof_xgb,oof_rf,oof_v14])
oof_meta = np.zeros(len(train))
for a,b in folds:
    lr = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr.fit(meta_X[a], y[a])
    oof_meta[b] = lr.predict_proba(meta_X[b])[:,1]
report(oof_meta,"stacked (LogReg meta)")

print(f"\nreference: single LightGBM baseline macroF1=0.6422")
log(f"total runtime: {(time.time()-T0)/60:.1f} min")
