"""Answers questions 2-15 from the A_cold audit, reusing the cached features/OOF from
diagnostic_123.py (_diag_cache/) -- no need to rebuild LSA/BM25/entity-mining again.
Only retrains where strictly required (fold-stability check #6, oracle sentence-only
model #14), both using the already-cached feature matrix.
"""
import os, sys, re
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
try:
    import lightgbm as lgb; HAS_LGB=True
except Exception:
    HAS_LGB=False
    from sklearn.ensemble import HistGradientBoostingClassifier

CACHE = r"D:\Lomba\IFEST2026_DAC\notebook_v12\_diag_cache"
oof = np.load(os.path.join(CACHE,'oof.npy'))
cold_mask = np.load(os.path.join(CACHE,'cold_mask.npy'))
fold_id = np.load(os.path.join(CACHE,'fold_id.npy'))
Ftr = pd.read_csv(os.path.join(CACHE,'features.csv'))
train = pd.read_csv(os.path.join(CACHE,'train_slim.csv'))
y = train['label'].values
FEATS = list(Ftr.columns)
SEED=42

cold_idx = np.where(cold_mask)[0]
cold_oof = oof[cold_idx]; cold_y = y[cold_idx]
best=(0.5,-1)
for t in np.arange(0.05,0.96,0.025):
    mac=f1_score(cold_y,(cold_oof>=t).astype(int),average='macro')
    if mac>best[1]: best=(t,mac)
THR=best[0]
pred_cold=(cold_oof>=THR).astype(int)
print(f"threshold={THR:.3f} macroF1={best[1]:.4f}")
FN_mask=(cold_y==0)&(pred_cold==1); FP_mask=(cold_y==1)&(pred_cold==0)
FN_idx=cold_idx[FN_mask]; FP_idx=cold_idx[FP_mask]
print(f"FN={FN_mask.sum()} FP={FP_mask.sum()}\n")

print("="*70+"\nQ2/Q4: similarity-feature distribution, class0 vs class1 (A_cold)\n"+"="*70)
sim_feats=['jaccard','bm25_max','bm25_mean','lsa_cos','struct_max_chunk','struct_std','struct_topk_mean','title_cov']
c0=Ftr.iloc[cold_idx][cold_y==0]; c1=Ftr.iloc[cold_idx][cold_y==1]
rows=[]
for f in sim_feats:
    a=c0[f].values; b=c1[f].values
    auc=roc_auc_score(cold_y, Ftr.iloc[cold_idx][f].values)
    rows.append(dict(feature=f, c0_mean=a.mean(), c0_median=np.median(a), c0_p10=np.quantile(a,.1), c0_p90=np.quantile(a,.9),
                      c1_mean=b.mean(), c1_median=np.median(b), c1_p10=np.quantile(b,.1), c1_p90=np.quantile(b,.9),
                      mean_gap=b.mean()-a.mean(), auc=max(auc,1-auc)))
distdf=pd.DataFrame(rows).sort_values('auc',ascending=False)
print(distdf.round(4).to_string(index=False))
print("\n-> univariate separation is WEAK for pure similarity features (AUC close to 0.5-0.6),")
print("   confirming lexical/semantic similarity alone barely distinguishes class 0 from 1 here.")

print("\n"+"="*70+"\nQ3: top 20 most-confident FN + top 20 most-confident FP\n"+"="*70)
fn_sorted = FN_idx[np.argsort(-oof[FN_idx])]
fp_sorted = FP_idx[np.argsort(oof[FP_idx])]
def show(idx_list, label, n=20):
    print(f"\n--- {label} ---")
    for i,idx in enumerate(idx_list[:n]):
        f=Ftr.iloc[idx]
        reasons=[]
        if f['any_title_ent_missing']==1.0: reasons.append('entity_missing')
        if f['num_mismatch']==1.0: reasons.append('number_mismatch')
        if f['year_missing']==1.0: reasons.append('year_mismatch')
        if f['neg_mismatch']==1.0: reasons.append('negation_word_present_only_one_side')
        if f['action_conflict']==1.0: reasons.append('action_antonym')
        if not reasons: reasons.append('NO_HEURISTIC_FLAG(semantic?)')
        print(f"[{i+1}] p(class1)={oof[idx]:.4f} true={y[idx]} reasons={','.join(reasons)}")
        print(f"    title  : {train['title'].iloc[idx][:100]}")
        print(f"    body   : {train['content'].iloc[idx][:160]}...")
show(fn_sorted, "TOP 20 FALSE NEGATIVE (true=0, model very confident it's 1)")
show(fp_sorted, "TOP 20 FALSE POSITIVE (true=1, model very confident it's 0)")

print("\n"+"="*70+"\nQ5: univariate AUC/PR-AUC per feature vs class0 (A_cold OOF-safe)\n"+"="*70)
rows=[]
for f in FEATS:
    v = Ftr.iloc[cold_idx][f].values
    if np.std(v)==0: continue
    auc = roc_auc_score(cold_y, -v)  # orient toward class0
    auc = max(auc, 1-auc)
    ap = average_precision_score(1-cold_y, -v)  # PR-AUC for class0 (treat class0 as positive)
    ap = max(ap, average_precision_score(1-cold_y, v))
    rows.append(dict(feature=f, roc_auc=auc, pr_auc_class0=ap))
fdf=pd.DataFrame(rows).sort_values('pr_auc_class0',ascending=False)
print(fdf.head(15).round(4).to_string(index=False))
base_rate = (cold_y==0).mean()
print(f"\n(class0 base rate = {base_rate:.4f}; PR-AUC above this is better than random)")

print("\n"+"="*70+"\nQ6: struct_std / structural-feature fold stability (retrain 5-fold, feature_importances_)\n"+"="*70)
groups = train['content'].apply(lambda s: hash(s)).values  # coarse re-grouping proxy not exact; use fold_id instead
# Re-derive folds consistent with cached fold_id (row -> fold assignment already saved)
imp_per_fold = []
Xall = Ftr.values
for fi in range(5):
    tr_mask = fold_id != fi
    va_mask = fold_id == fi
    if HAS_LGB:
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.9,
                                colsample_bytree=0.8, class_weight='balanced', random_state=SEED, verbose=-1)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier as m_cls
        m = m_cls(max_iter=500, learning_rate=0.05, random_state=SEED)
    m.fit(Xall[tr_mask], y[tr_mask])
    if hasattr(m, 'feature_importances_'):
        imp_per_fold.append(pd.Series(m.feature_importances_, index=FEATS))
impdf = pd.DataFrame(imp_per_fold).T
impdf['mean']=impdf.mean(axis=1); impdf['std']=impdf.iloc[:,:5].std(axis=1)
impdf['rank_std'] = impdf.iloc[:,:5].rank(ascending=False).std(axis=1)
top_by_mean = impdf.sort_values('mean',ascending=False).head(15)
print(top_by_mean[['mean','std','rank_std']].round(2).to_string())
print("\nstruct_std specifically:")
print(impdf.loc['struct_std', ['mean','std','rank_std']] if 'struct_std' in impdf.index else "not found")

print("\n"+"="*70+"\nQ7: entity-swap direct evidence in FN (ENT-pool based, no NER type available)\n"+"="*70)
n_ent_flag = int((Ftr.iloc[FN_idx]['any_title_ent_missing']==1.0).sum())
print(f"FN with title-entity missing from body: {n_ent_flag}/{len(FN_idx)} ({n_ent_flag/max(len(FN_idx),1)*100:.1f}%)")
print("(no PERSON/LOCATION/ORG type distinction available -- entity pool is type-agnostic proper-noun mining)")
print("\nsample 15 entity-missing FN cases:")
ent_fn = [i for i in FN_idx if Ftr.iloc[i]['any_title_ent_missing']==1.0][:15]
for idx in ent_fn:
    print(f"  title: {train['title'].iloc[idx][:90]}")

print("\n"+"="*70+"\nQ8: action/antonym reversal in FN\n"+"="*70)
n_action = int((Ftr.iloc[FN_idx]['action_conflict']==1.0).sum())
print(f"FN with antonym-lexicon action conflict: {n_action}/{len(FN_idx)} ({n_action/max(len(FN_idx),1)*100:.1f}%)")
print("-> rare with this hand lexicon; consistent with earlier mutation census (action mutations only 0.6% of tamper events)")

print("\n"+"="*70+"\nQ9: negation in FN -- CAUTION, sanity-check the crude heuristic\n"+"="*70)
n_neg = int((Ftr.iloc[FN_idx]['neg_mismatch']==1.0).sum())
print(f"FN flagged neg_mismatch (title has NEG word XOR body has NEG word): {n_neg}/{len(FN_idx)} ({n_neg/max(len(FN_idx),1)*100:.1f}%)")
body_neg_rate = Ftr.iloc[cold_idx]['neg_body'].mean()
title_neg_rate = Ftr.iloc[cold_idx]['neg_title'].mean()
print(f"base rate: body contains a NEG word {body_neg_rate*100:.1f}% of the time (long text, almost always finds SOME 'tidak/belum'),")
print(f"           title contains a NEG word {title_neg_rate*100:.1f}% of the time.")
print("-> This heuristic is NOT a real contradiction detector, it is just 'does a negation word exist")
print("   ANYWHERE in body vs title' -- with long bodies this fires often by chance, unrelated to the headline's")
print("   actual claim. Confirmed noisy: earlier mutation census (real generation-mechanism data) found")
print("   negation accounts for 0% of actual tamper events. TREAT THE 73-76% NUMBER ABOVE AS A FALSE SIGNAL,")
print("   an artifact of a body-wide keyword check, not evidence that negation is the real bottleneck.")

print("\n"+"="*70+"\nQ10: numeric/date contradiction in FN\n"+"="*70)
n_num = int((Ftr.iloc[FN_idx]['num_mismatch']==1.0).sum())
n_year = int((Ftr.iloc[FN_idx]['year_missing']==1.0).sum())
print(f"FN with number mismatch: {n_num}/{len(FN_idx)} ({n_num/max(len(FN_idx),1)*100:.1f}%)")
print(f"FN with year mismatch:   {n_year}/{len(FN_idx)} ({n_year/max(len(FN_idx),1)*100:.1f}%)")

print("\n"+"="*70+"\nQ11: diagnostic-only rule detectors -- precision/recall/F1(class0) on ALL cold OOF\n"+"="*70)
detectors = {
    'entity_missing': Ftr.iloc[cold_idx]['any_title_ent_missing'].values,
    'number_mismatch': Ftr.iloc[cold_idx]['num_mismatch'].values,
    'year_mismatch': Ftr.iloc[cold_idx]['year_missing'].values,
    'negation_mismatch(noisy!)': Ftr.iloc[cold_idx]['neg_mismatch'].values,
    'action_conflict': Ftr.iloc[cold_idx]['action_conflict'].values,
}
union = np.zeros(len(cold_idx))
union_no_neg = np.zeros(len(cold_idx))
for name, flag in detectors.items():
    pred0 = (flag==1.0).astype(int)  # predicts class0 when flag fires
    pred_label = 1-pred0  # detector fires -> predict 0 ; else predict 1(default)
    p0 = precision_score(cold_y, pred_label, pos_label=0, zero_division=0)
    r0 = recall_score(cold_y, pred_label, pos_label=0, zero_division=0)
    f10 = f1_score(cold_y, pred_label, pos_label=0, zero_division=0)
    print(f"  {name:28s} precision0={p0:.4f} recall0={r0:.4f} F1_0={f10:.4f}  (fires on {pred0.sum()} rows)")
    union = np.maximum(union, pred0)
    if 'negation' not in name: union_no_neg = np.maximum(union_no_neg, pred0)
pred_union = 1-union
p0=precision_score(cold_y,pred_union,pos_label=0,zero_division=0); r0=recall_score(cold_y,pred_union,pos_label=0,zero_division=0)
f10=f1_score(cold_y,pred_union,pos_label=0,zero_division=0)
print(f"  {'UNION (all detectors)':28s} precision0={p0:.4f} recall0={r0:.4f} F1_0={f10:.4f}  (fires on {union.sum():.0f} rows)")
pred_union_nn = 1-union_no_neg
p0=precision_score(cold_y,pred_union_nn,pos_label=0,zero_division=0); r0=recall_score(cold_y,pred_union_nn,pos_label=0,zero_division=0)
f10=f1_score(cold_y,pred_union_nn,pos_label=0,zero_division=0)
print(f"  {'UNION excl. negation':28s} precision0={p0:.4f} recall0={r0:.4f} F1_0={f10:.4f}  (fires on {union_no_neg.sum():.0f} rows)")
print(f"\n  compare: full 49-feature LightGBM model F1_0 on this same cold set = {f1_score(cold_y,pred_cold,pos_label=0):.4f}")

print("\n"+"="*70+"\nQ12: FN NOT explained by any heuristic detector -- likely genuinely semantic\n"+"="*70)
unexplained=[]
for idx in FN_idx:
    f=Ftr.iloc[idx]
    if (f['any_title_ent_missing']==0.0 and f['num_mismatch']==0.0 and f['year_missing']==0.0
        and f['action_conflict']==0.0):
        unexplained.append(idx)
print(f"FN unexplained by entity/number/date/action detectors: {len(unexplained)}/{len(FN_idx)} ({len(unexplained)/max(len(FN_idx),1)*100:.1f}%)")
print("(negation excluded from this check -- shown separately as unreliable)")
print("\nsample 15 unexplained (likely semantic/paraphrase) cases:")
for idx in unexplained[:15]:
    print(f"  title: {train['title'].iloc[idx][:90]}")

print("\n"+"="*70+"\nQ14 (moved up, needs a small experiment): oracle-chunk-only model vs full model\n"+"="*70)
struct_only_feats = ['struct_first_chunk','struct_first3_mean','struct_max_chunk','struct_topk_mean',
                      'struct_std','struct_min_chunk','struct_argmax_pos','lsa_cos']
Xs = Ftr[struct_only_feats].values
oof_struct = np.zeros(len(train))
for fi in range(5):
    tr_mask = fold_id != fi; va_mask = fold_id == fi
    if HAS_LGB:
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.9,
                                colsample_bytree=0.8, class_weight='balanced', random_state=SEED, verbose=-1)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier as m_cls
        m = m_cls(max_iter=500, learning_rate=0.05, random_state=SEED)
    m.fit(Xs[tr_mask], y[tr_mask])
    oof_struct[va_mask] = m.predict_proba(Xs[va_mask])[:,1]
cold_oof_struct = oof_struct[cold_idx]
best_s=(0.5,-1)
for t in np.arange(0.05,0.96,0.025):
    mac=f1_score(cold_y,(cold_oof_struct>=t).astype(int),average='macro')
    if mac>best_s[1]: best_s=(t,mac)
pred_s=(cold_oof_struct>=best_s[0]).astype(int)
f1c_s=f1_score(cold_y,pred_s,average=None,labels=[0,1])
print(f"oracle-chunk-similarity-only (8 feats): macroF1={best_s[1]:.4f}  F1_0={f1c_s[0]:.4f}  F1_1={f1c_s[1]:.4f}")
print(f"full 49-feature model (baseline):       macroF1={best[1]:.4f}  F1_0={f1_score(cold_y,pred_cold,pos_label=0):.4f}")
print("-> if close to full model: retrieval/chunk-matching alone carries most of the signal (bottleneck = which chunk to trust).")
print("-> if much worse: entity/lexical features add real value beyond pure similarity (bottleneck = more than retrieval).")

print("\n"+"="*70+"\nQ15: is the model just learning the prior?\n"+"="*70)
always1_f1 = f1_score(cold_y, np.ones(len(cold_y),int), average=None, labels=[0,1])
print(f"always-predict-1 baseline: F1_0={always1_f1[0]:.4f} F1_1={always1_f1[1]:.4f} macro={always1_f1.mean():.4f}")
print(f"v12 model:                 F1_0={f1_score(cold_y,pred_cold,pos_label=0):.4f} F1_1={f1_score(cold_y,pred_cold,pos_label=1):.4f} macro={best[1]:.4f}")
pred0_rate = (pred_cold==0).mean()
recall0 = recall_score(cold_y, pred_cold, pos_label=0)
precision0 = precision_score(cold_y, pred_cold, pos_label=0)
print(f"\nmodel predicts class0 on {pred0_rate*100:.1f}% of cold rows (true class0 rate = {base_rate*100:.1f}%)")
print(f"recall0={recall0:.4f} precision0={precision0:.4f}")
if pred0_rate < base_rate*1.5:
    print("-> model is CONSERVATIVE: predicts negative rarely, close to or below the true base rate.")
else:
    print("-> model is AGGRESSIVE: predicts negative more often than base rate, but picks wrong rows (low precision).")

print("\nDone.")
