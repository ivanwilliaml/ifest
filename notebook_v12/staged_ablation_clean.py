"""Properly isolated staged ablation for A_cold, starting from the TRUE original v12
48-feature baseline (action_conflict was accidentally baked into the cache during
diagnostics and contaminated the last two experiments -- removed here).

Exp A: true_baseline + action_conflict (antonym lexicon) ALONE
Exp B: true_baseline + refined number-consistency (continuous, not just boolean) ALONE
Exp C: true_baseline + entity_role_conflict (cleaned pool) ALONE
Exp D: true_baseline + winners combined
Reports Macro F1, F1-0, F1-1, Precision-0, Recall-0 for every stage, plus per-row
FN/FP transition counts (how many rows flip label vs baseline) for the entity-role feature.
"""
import os, sys, re, string as _s, unicodedata, hashlib
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
Ftr_full = pd.read_csv(os.path.join(CACHE,'features.csv'))
train = pd.read_csv(os.path.join(CACHE,'train_slim.csv'))
y = train['label'].values
SEED=42

TRUE_V12_COLS = [c for c in Ftr_full.columns if c != 'action_conflict']  # 48 original cols
Ftr_base = Ftr_full[TRUE_V12_COLS].copy()
print(f"true baseline columns: {len(TRUE_V12_COLS)}")

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
print(f"cold rows: {cold_mask.sum()}")

groups=train['ch'].values
sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
folds=list(sgkf.split(train,y,groups))

# ---------- feature builders ----------
NUM=re.compile(r'\d+[.,]?\d*\s*%?')
ANTONYMS=[('naik','turun'),('mulai','tunda'),('setuju','tolak'),('dukung','lawan'),
          ('izinkan','larang'),('terima','tolak'),('buka','tutup'),('lanjut','hentikan'),
          ('terapkan','tunda'),('naikkan','turunkan'),('tambah','kurang')]
def action_conflict_feat(title, body):
    tset=set(title.lower().split()); cset=set(body.lower().split())
    for w1,w2 in ANTONYMS:
        if (w1 in tset and w2 in cset) or (w2 in tset and w1 in cset): return 1.0
    return 0.0

def parse_num(s):
    s=s.replace('%','').replace(',','.')
    try: return float(s)
    except: return None

def number_refined_feats(title, body):
    tn = NUM.findall(title); bn = NUM.findall(body)
    tvals = [parse_num(x) for x in tn]; tvals=[v for v in tvals if v is not None]
    bvals = [parse_num(x) for x in bn]; bvals=[v for v in bvals if v is not None]
    if not tvals:
        return dict(num_unmatched_frac=0.0, num_max_rel_gap=0.0, num_has_close_but_not_exact=0.0)
    bset_str = set(bn)
    unmatched = [v for v,s in zip(tvals,tn) if s not in bset_str]
    frac = len(unmatched)/len(tvals)
    max_gap = 0.0; close_not_exact = 0.0
    for v in unmatched:
        if not bvals: continue
        nearest = min(bvals, key=lambda b: abs(b-v))
        rel_gap = abs(nearest-v)/max(abs(v),1e-6)
        max_gap = max(max_gap, rel_gap)
        if rel_gap < 0.5: close_not_exact = 1.0  # a plausible near-value swap, e.g. 5.2% vs 4.2%
    return dict(num_unmatched_frac=frac, num_max_rel_gap=min(max_gap,10.0), num_has_close_but_not_exact=close_not_exact)

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
def entity_role_conflict_feat(title, body):
    Ttok=title.split()
    tents=[w for w in Ttok if w in ENT_CLEAN]
    craw={w.strip(_s.punctuation) for w in body.split()}
    missing=[w for w in tents if w not in craw]
    if not missing: return 0.0
    tc = top_chunk(title, chunks_of(body))
    chunk_ents=[w.strip(_s.punctuation) for w in tc.split() if w.strip(_s.punctuation) in ENT_CLEAN]
    for m in missing:
        if any(e!=m for e in chunk_ents): return 1.0
    return 0.0

print("computing action_conflict...")
Ftr_base['action_conflict'] = [action_conflict_feat(t,b) for t,b in zip(titles,bodies)]
print("computing number_refined...")
numfeats = [number_refined_feats(t,b) for t,b in zip(titles,bodies)]
numdf = pd.DataFrame(numfeats)
print("computing entity_role_conflict...")
Ftr_base['entity_role_conflict'] = [entity_role_conflict_feat(t,b) for t,b in zip(titles,bodies)]

FEAT_ACTION = TRUE_V12_COLS + ['action_conflict']
FEAT_NUMBER = TRUE_V12_COLS + list(numdf.columns)
FEAT_ENTITY = TRUE_V12_COLS + ['entity_role_conflict']
Ftr_number = pd.concat([Ftr_base[TRUE_V12_COLS], numdf], axis=1)

def run_cv(Xdf):
    oof=np.zeros(len(train))
    for a,b in folds:
        Xa=Xdf.iloc[a].values; Xb=Xdf.iloc[b].values
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
    return dict(name=name, macro=best[1], f1_0=f1c[0], f1_1=f1c[1], p0=p0, r0=r0, pred=pred, cold_idx=cold_idx, cy=cy)

print("\n=== TRUE baseline (48 original v12 features) ===")
oof_true_base = run_cv(Ftr_base[TRUE_V12_COLS])
r_base = report(oof_true_base, "true baseline (48 feats)")

print("\n=== Exp A: + action_conflict ONLY ===")
oof_a = run_cv(Ftr_base[FEAT_ACTION])
r_a = report(oof_a, "+ action_conflict (49 feats)")

print("\n=== Exp B: + number_refined ONLY ===")
oof_b = run_cv(Ftr_number)
r_b = report(oof_b, "+ number_refined (51 feats)")

print("\n=== Exp C: + entity_role_conflict ONLY ===")
oof_c = run_cv(Ftr_base[FEAT_ENTITY])
r_c = report(oof_c, "+ entity_role_conflict (49 feats)")

print("\n=== FN/FP transition analysis for entity_role_conflict (vs true baseline) ===")
base_pred_full = r_base['pred']; c_pred_full = r_c['pred']
cy = r_base['cy']
base_FN = (cy==0)&(base_pred_full==1); c_FN=(cy==0)&(c_pred_full==1)
base_FP = (cy==1)&(base_pred_full==0); c_FP=(cy==1)&(c_pred_full==0)
fixed_FN = base_FN & (~c_FN)   # was FN, now correct
new_FP   = (~base_FP) & c_FP   # was correct, now FP
still_FN = base_FN & c_FN
print(f"baseline FN={base_FN.sum()}  ->  entity-role FN={c_FN.sum()}")
print(f"  FN fixed (1->0 correctly): {fixed_FN.sum()}")
print(f"  new FP introduced (1 wrongly flagged 0): {new_FP.sum()}")
print(f"  still FN (unchanged): {still_FN.sum()}")

print("\n=== Exp D: combine winners (macro-gain > 0 stages) ===")
winners = []
for r, cols in [(r_a, FEAT_ACTION), (r_b, list(Ftr_number.columns)), (r_c, FEAT_ENTITY)]:
    if r['macro'] > r_base['macro']:
        winners.append((r['name'], cols))
print(f"winners (macro > baseline {r_base['macro']:.4f}): {[w[0] for w in winners]}")
if len(winners) >= 2:
    combo_cols = list(TRUE_V12_COLS)
    if any('action' in n for n,_ in winners): combo_cols += ['action_conflict']
    if any('number' in n for n,_ in winners): combo_cols += list(numdf.columns)
    if any('entity' in n for n,_ in winners): combo_cols += ['entity_role_conflict']
    combo_df = pd.concat([Ftr_base[TRUE_V12_COLS], Ftr_base[['action_conflict']] if 'action_conflict' in combo_cols else pd.DataFrame(index=Ftr_base.index),
                           numdf if any('num_' in c for c in combo_cols) else pd.DataFrame(index=Ftr_base.index),
                           Ftr_base[['entity_role_conflict']] if 'entity_role_conflict' in combo_cols else pd.DataFrame(index=Ftr_base.index)], axis=1)
    combo_df = combo_df.loc[:,~combo_df.columns.duplicated()]
    oof_d = run_cv(combo_df)
    report(oof_d, f"combined winners ({combo_df.shape[1]} feats)")
else:
    print("fewer than 2 winners -- combination skipped, not enough signal to combine meaningfully")

print(f"\nreference: true baseline macroF1={r_base['macro']:.4f} F1_0={r_base['f1_0']:.4f}")
