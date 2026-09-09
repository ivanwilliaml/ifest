"""Deep diagnostic (no modeling) for A_cold. Core 5 questions:
Q1: empirical corruption-mechanism distribution among the 483 cold negatives (multi-signal
    categorization + printed audit samples, not a single boolean detector).
Q2: true FN error taxonomy for the top-confidence false negatives.
Q5: can cold negatives be traced to a nearest positive title elsewhere in the corpus?
Q7: OOF probability distribution for class-0 rows (FN vs TP), quantiles + bucket counts.
Q15: oracle ceiling F1-0 if a perfect (precision=1) detector caught every row with an
     "obvious" entity/action/number/date conflict signal.
"""
import os, sys, re, string as _s, unicodedata
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

CACHE = r"D:\Lomba\IFEST2026_DAC\notebook_v12\_diag_cache"
Ftr = pd.read_csv(os.path.join(CACHE,'features.csv'))
train = pd.read_csv(os.path.join(CACHE,'train_slim.csv'))
oof = np.load(os.path.join(CACHE,'oof.npy'))
cold_mask = np.load(os.path.join(CACHE,'cold_mask.npy'))
y = train['label'].values
cold_idx = np.where(cold_mask)[0]
cy = y[cold_idx]; co = oof[cold_idx]

best=(0.5,-1)
for t in np.arange(0.05,0.96,0.025):
    mac=f1_score(cy,(co>=t).astype(int),average='macro')
    if mac>best[1]: best=(t,mac)
THR=best[0]
pred_cold=(co>=THR).astype(int)
print(f"threshold={THR:.3f} macroF1={best[1]:.4f}")

neg_idx = cold_idx[cy==0]           # all 483ish true cold negatives
FN_idx = cold_idx[(cy==0)&(pred_cold==1)]
TP0_idx = cold_idx[(cy==0)&(pred_cold==0)]
print(f"total cold negatives: {len(neg_idx)}  FN: {len(FN_idx)}  correctly-caught TP0: {len(TP0_idx)}")

# =====================================================================
print("\n"+"="*70+"\nQ1: corruption-mechanism distribution among ALL 483 cold negatives\n"+"="*70)
# =====================================================================
def categorize(idx):
    f = Ftr.iloc[idx]
    T = train['title'].iloc[idx]; C = train['content'].iloc[idx]
    if f['any_title_ent_missing']==1.0 and f.get('action_conflict',0)==0:
        return 'entity_substitution'
    if f.get('action_conflict',0)==1.0:
        return 'action_reversal'
    if f['num_mismatch']==1.0:
        return 'number_substitution'
    if f['year_missing']==1.0:
        return 'date_substitution'
    if f['lsa_cos']<0.25 and f['jaccard']<0.03:
        return 'different_event(low_overlap)'
    return 'semantic_or_undetermined'

cats = Counter()
examples = {}
for idx in neg_idx:
    c = categorize(idx)
    cats[c]+=1
    examples.setdefault(c, []).append(idx)
total = len(neg_idx)
for c,n in cats.most_common():
    print(f"  {c:32s} n={n:4d} ({n/total*100:5.1f}%)")

print("\n--- audit samples (5 per category) ---")
for c in cats:
    print(f"\n[{c}]")
    for idx in examples[c][:5]:
        print(f"  T: {train['title'].iloc[idx][:90]}")
        print(f"  B: {train['content'].iloc[idx][:150]}...")

# =====================================================================
print("\n"+"="*70+"\nQ2: TRUE error taxonomy for top-150 highest-confidence FN\n"+"="*70)
# =====================================================================
fn_sorted = FN_idx[np.argsort(-oof[FN_idx])]
top_fn = fn_sorted[:150]
fn_cats = Counter(); fn_examples={}
for idx in top_fn:
    c = categorize(idx)
    fn_cats[c]+=1
    fn_examples.setdefault(c,[]).append(idx)
for c,n in fn_cats.most_common():
    print(f"  {c:32s} n={n:4d} ({n/len(top_fn)*100:5.1f}%)")
print("\n--- audit samples (5 per category, top-confidence FN) ---")
for c in fn_cats:
    print(f"\n[{c}]")
    for idx in fn_examples[c][:5]:
        print(f"  p(class1)={oof[idx]:.4f}")
        print(f"  T: {train['title'].iloc[idx][:90]}")
        print(f"  B: {train['content'].iloc[idx][:150]}...")

# =====================================================================
print("\n"+"="*70+"\nQ5: can cold negatives be traced to a nearest positive title elsewhere?\n"+"="*70)
# =====================================================================
titles_all = train['title'].fillna('').tolist()
word_tv = TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True, max_features=30000)
Wm = word_tv.fit_transform(titles_all)
Wm = Wm.multiply(1.0/np.maximum(np.sqrt(Wm.multiply(Wm).sum(axis=1)),1e-9)).tocsr()
pos_idx_all = np.where(y==1)[0]
Wpos = Wm[pos_idx_all]

nearest_sims=[]; nearest_titles=[]
for idx in neg_idx:
    sims = (Wm[idx] @ Wpos.T).toarray().ravel()
    j = np.argmax(sims)
    nearest_sims.append(sims[j]); nearest_titles.append(pos_idx_all[j])
nearest_sims = np.array(nearest_sims)
for thr in [0.9,0.8,0.7,0.6,0.5]:
    frac = (nearest_sims>=thr).mean()
    print(f"  nearest-positive-title cosine >= {thr}: {frac*100:.1f}% of cold negatives")
print(f"  mean nearest-positive-title cosine: {nearest_sims.mean():.4f}  median: {np.median(nearest_sims):.4f}")

print("\n--- 10 highest-similarity negative->nearest-positive pairs (likely traceable mutations) ---")
order = np.argsort(-nearest_sims)[:10]
for k in order:
    idx = neg_idx[k]; src = nearest_titles[k]
    print(f"  sim={nearest_sims[k]:.4f}")
    print(f"    NEGATIVE title: {train['title'].iloc[idx][:90]}")
    print(f"    NEAREST POS title (diff article): {train['title'].iloc[src][:90]}")

# =====================================================================
print("\n"+"="*70+"\nQ7: OOF probability distribution, class-0 rows (FN vs correctly-caught)\n"+"="*70)
# =====================================================================
p_neg_all = oof[neg_idx]
qs=[0,10,25,50,75,90,100]
print("class0 (all 483) p(class1) quantiles:", {q: round(float(np.percentile(p_neg_all,q)),4) for q in qs})
p_fn = oof[FN_idx]
p_tp0 = oof[TP0_idx]
print("FN-only p(class1) quantiles:         ", {q: round(float(np.percentile(p_fn,q)),4) for q in qs})
print("correctly-caught-negative p(class1):  ", {q: round(float(np.percentile(p_tp0,q)),4) for q in qs})
print("\nbucket counts (class0, all 483):")
edges=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01]
hist,_=np.histogram(p_neg_all, bins=edges)
for i in range(len(hist)):
    print(f"  [{edges[i]:.1f},{edges[i+1]:.1f}): {hist[i]:4d} ({hist[i]/len(p_neg_all)*100:5.1f}%)")
print(f"\nFN with p(class1) > 0.9: {(p_fn>0.9).sum()}/{len(p_fn)} ({(p_fn>0.9).mean()*100:.1f}%) -- model VERY confidently wrong")
print(f"FN with p(class1) in [0.4,0.6] (near threshold, genuinely uncertain): {((p_fn>=0.4)&(p_fn<=0.6)).sum()}/{len(p_fn)} ({((p_fn>=0.4)&(p_fn<=0.6)).mean()*100:.1f}%)")

# =====================================================================
print("\n"+"="*70+"\nQ15: oracle ceiling -- perfect detector on 'obvious conflict' union\n"+"="*70)
# =====================================================================
obvious = (Ftr.iloc[neg_idx]['any_title_ent_missing']==1.0) | (Ftr.iloc[neg_idx]['num_mismatch']==1.0) | \
          (Ftr.iloc[neg_idx].get('action_conflict',0)==1.0) | (Ftr.iloc[neg_idx]['year_missing']==1.0)
recall_oracle = obvious.mean()
f1_0_oracle = 2*recall_oracle/(1+recall_oracle)  # precision=1 assumed (true oracle, never misfires)
print(f"fraction of 483 cold negatives with >=1 obvious entity/action/number/date signal: {recall_oracle*100:.1f}%")
print(f"oracle F1-0 ceiling (perfect precision=1, recall=this fraction): {f1_0_oracle:.4f}")
print(f"current real model F1-0: 0.3241-0.3540 (depending on config)")
print(f"gap: oracle ceiling is {f1_0_oracle:.4f} vs real ~0.33-0.35 -> headroom = {f1_0_oracle-0.34:.4f}")
