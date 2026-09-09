"""Hyperparameter sweep for the LightGBM dense model on A_cold, using the same 48
true-baseline v12 features + refined entity_role_conflict (net-neutral but keeps the
strongest available feature set). Grouped 5-fold CV, same methodology as before.
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score
import lightgbm as lgb

SEED=42
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

CACHE = r"D:\Lomba\IFEST2026_DAC\notebook_v12\_diag_cache"
Ftr = pd.read_csv(os.path.join(CACHE,'features.csv'))
train = pd.read_csv(os.path.join(CACHE,'train_slim.csv'))
y = train['label'].values
FEAT_COLS = [c for c in Ftr.columns if c != 'action_conflict']  # true 48-feature baseline

import hashlib, re, unicodedata
def nh(x):
    x=unicodedata.normalize('NFKC',str(x)).lower(); return re.sub(r'\s+',' ',x).strip()
train['ch']=train['content'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
_cnt=train['ch'].value_counts()
cold_mask = train['ch'].map(_cnt).eq(1).values
log(f"cold rows: {cold_mask.sum()}")

groups=train['ch'].values
sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
folds=list(sgkf.split(train,y,groups))
X = Ftr[FEAT_COLS].values

def run_cv(params):
    oof=np.zeros(len(train))
    for a,b in folds:
        m=lgb.LGBMClassifier(random_state=SEED, verbose=-1, **params)
        m.fit(X[a],y[a])
        oof[b]=m.predict_proba(X[b])[:,1]
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
    print(f"{name:55s} macroF1={best[1]:.4f} F1_0={f1c[0]:.4f} F1_1={f1c[1]:.4f} P0={p0:.4f} R0={r0:.4f}")
    return best[1]

configs = {
    'current (n=600,leaves=31,lr=.05)':      dict(n_estimators=600, num_leaves=31, learning_rate=0.05,
                                                    subsample=0.9, colsample_bytree=0.8, class_weight='balanced'),
    'bigger (n=3000,leaves=63,lr=.02)':      dict(n_estimators=3000, num_leaves=63, learning_rate=0.02,
                                                    subsample=0.9, colsample_bytree=0.8, class_weight='balanced'),
    'huge (n=6000,leaves=127,lr=.01)':       dict(n_estimators=6000, num_leaves=127, learning_rate=0.01,
                                                    subsample=0.8, colsample_bytree=0.7, class_weight='balanced'),
    'deep+regularized (n=3000,leaves=63,lr=.02,reg)':
        dict(n_estimators=3000, num_leaves=63, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7,
             class_weight='balanced', reg_alpha=0.5, reg_lambda=1.0, min_child_samples=20),
    'shallow+wide (n=2000,leaves=15,lr=.03)': dict(n_estimators=2000, num_leaves=15, learning_rate=0.03,
                                                     subsample=0.9, colsample_bytree=0.8, class_weight='balanced'),
    'small (n=200,leaves=15,lr=.1)':          dict(n_estimators=200, num_leaves=15, learning_rate=0.1,
                                                     subsample=0.9, colsample_bytree=0.8, class_weight='balanced'),
}

results=[]
for name, params in configs.items():
    log(f"=== {name} ===")
    oof = run_cv(params)
    mac = report(oof, name)
    results.append((name, mac))

print("\n=== SUMMARY (sorted) ===")
for name, mac in sorted(results, key=lambda r: -r[1]):
    print(f"  {name:55s} macroF1={mac:.4f}")
print(f"\nreference: true baseline (48 feats, original config) macroF1=0.6422")
log(f"total runtime: {(time.time()-T0)/60:.1f} min")
