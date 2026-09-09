"""Phased feature ablation specifically for B_no_pos (body seen, no positive title in
memory). Baseline hard rule always predicts 1 here; goal is to see how far a
contradiction-aware classifier can push F1-0 without touching A/cold-start.
All leave-one-title-out per body (memory excludes only the row itself).
"""
import re, time, hashlib, unicodedata, string as _s
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold
try:
    import lightgbm as lgb; HAS_LGB=True
except Exception:
    HAS_LGB=False
    from sklearn.ensemble import HistGradientBoostingClassifier

SEED=42
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

train = pd.read_csv(r"D:\Lomba\IFEST2026_DAC\data\train.csv")
def nh(x):
    x=unicodedata.normalize('NFKC',str(x)).lower(); return re.sub(r'\s+',' ',x).strip()
train['ch']=train['content'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
train['th']=train['title'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
URL=re.compile(r'https?://\S+|www\.\S+'); HTML=re.compile(r'<[^>]+>'); WS=re.compile(r'\s+')
def clean(s):
    s=unicodedata.normalize('NFKC',str(s)); s=HTML.sub(' ',s); s=URL.sub(' ',s)
    return WS.sub(' ',s).strip()
train['T']=train['title'].apply(clean); train['C']=train['content'].apply(clean)

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

word_tv = TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True, max_features=20000)
char_tv = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=1, sublinear_tf=True, max_features=20000)
Wm = word_tv.fit_transform(train['T']); Cm = char_tv.fit_transform(train['T'])
Wm = Wm.multiply(1.0/np.maximum(np.sqrt(Wm.multiply(Wm).sum(axis=1)),1e-9)).tocsr()
Cm = Cm.multiply(1.0/np.maximum(np.sqrt(Cm.multiply(Cm).sum(axis=1)),1e-9)).tocsr()
log(f"title TF-IDF built: word={Wm.shape} char={Cm.shape}")

by_body = defaultdict(list)
titles = train['T'].tolist(); bodies = train['C'].tolist(); labels = train['label'].values
for i, ch in enumerate(train['ch']): by_body[ch].append(i)

def edit_conflict(a_title, b_title):
    ta, tb = a_title.split(), b_title.split()
    sa, sb = set(ta), set(tb)
    ents_a = [w for w in ta if w in ENT]
    ent_conflict = float(any(w not in sb for w in ents_a) and bool(ents_a))
    na, nb = set(NUM.findall(a_title)), set(NUM.findall(b_title))
    num_conflict = float(bool(na) and na != nb)
    ya, yb = set(YEAR.findall(a_title)), set(YEAR.findall(b_title))
    year_conflict = float(bool(ya) and ya != yb)
    neg_a = any(w in sa for w in NEG); neg_b = any(w in sb for w in NEG)
    neg_conflict = float(neg_a != neg_b)
    action_conflict = 0.0
    for w1,w2 in ANTONYMS:
        if (w1 in sa and w2 in sb) or (w2 in sa and w1 in sb): action_conflict = 1.0; break
    added = len(sb-sa); removed = len(sa-sb); shared = len(sa&sb)
    shared_ratio = shared/max(len(sa|sb),1)
    return dict(ent_conflict=ent_conflict, num_conflict=num_conflict, year_conflict=year_conflict,
                neg_conflict=neg_conflict, action_conflict=action_conflict,
                added=added, removed=removed, shared_ratio=shared_ratio)

def body_evidence(title, body):
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
        body_jaccard=len(inter)/max(len(union),1), body_title_cov=len(inter)/max(len(tset),1),
        body_ent_missing=float(len(miss)>0), body_frac_ent_missing=len(miss)/max(len(tents),1),
        body_num_mismatch=float(bool(tn) and not (tn&cn)), body_year_missing=float(bool(ty) and not (ty&cy)),
        body_neg_mismatch=float(neg_t != neg_c), body_action_conflict=action_hit,
    )

log("collecting B_no_pos pool (leave-one-title-out, no positive in memory)...")
rows_feat = []
for ch, idxs in by_body.items():
    if len(idxs) < 2: continue
    distinct_titles = set(train['th'].iloc[idxs])
    if len(distinct_titles) < 2: continue
    for i in idxs:
        memory_idx = [j for j in idxs if j != i]
        has_pos = any(labels[j]==1 for j in memory_idx)
        if has_pos: continue  # only B_no_pos
        neg_idx = [j for j in memory_idx if labels[j]==0]
        if not neg_idx: continue  # need at least one negative to retrieve against
        wsims = np.array([float(Wm[i].multiply(Wm[j]).sum()) for j in neg_idx])
        csims = np.array([float(Cm[i].multiply(Cm[j]).sum()) for j in neg_idx])
        order = np.argsort(-wsims)
        best_j = neg_idx[order[0]]
        f = dict(
            neg_max_word=float(wsims.max()), neg_top3_word=float(np.sort(wsims)[::-1][:3].mean()),
            neg_max_char=float(csims.max()), neg_top3_char=float(np.sort(csims)[::-1][:3].mean()),
            neg_n=len(neg_idx), neg_mean_word=float(wsims.mean()),
        )
        # B1: edit/conflict vs nearest negative
        f.update({f'edit_{k}':v for k,v in edit_conflict(titles[i], titles[best_j]).items()})
        # B2: body evidence (title vs its own body directly)
        f.update(body_evidence(titles[i], bodies[i]))
        # B3: negative prototype / cluster
        if len(neg_idx) >= 2:
            pair_sims = []
            for a in range(len(neg_idx)):
                for b in range(a+1, len(neg_idx)):
                    pair_sims.append(float(Wm[neg_idx[a]].multiply(Wm[neg_idx[b]]).sum()))
            cohesion = float(np.mean(pair_sims))
        else:
            cohesion = 1.0
        f['neg_cluster_cohesion'] = cohesion
        f['neg_outlier_score'] = float(wsims.mean()) - cohesion
        f['neg_topk_consistency'] = float(np.std(wsims)) if len(wsims)>1 else 0.0
        f['label'] = labels[i]
        f['hard_rule_pred'] = 1  # B_no_pos rule always predicts 1
        rows_feat.append(f)
log(f"B_no_pos pool: {len(rows_feat)}")

FB = pd.DataFrame(rows_feat)
y_B = FB['label'].values
hard_pred = FB['hard_rule_pred'].values
all_cols = [c for c in FB.columns if c not in ('label','hard_rule_pred')]
retrieval_cols = ['neg_max_word','neg_top3_word','neg_max_char','neg_top3_char','neg_n','neg_mean_word']
edit_cols = [c for c in all_cols if c.startswith('edit_')]
body_cols = [c for c in all_cols if c.startswith('body_')]
proto_cols = ['neg_cluster_cohesion','neg_outlier_score','neg_topk_consistency']

def cv_eval(cols, class_weight=None):
    X = FB[cols].values
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_B))
    for tr_i, va_i in kf.split(X):
        cw = class_weight if class_weight is not None else 'balanced'
        if HAS_LGB:
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                    min_child_samples=5, class_weight=cw, random_state=SEED, verbose=-1)
        else:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)
        m.fit(X[tr_i], y_B[tr_i])
        oof[va_i] = m.predict_proba(X[va_i])[:,1]
    return oof

def report(name, y_true, y_pred, results):
    acc = accuracy_score(y_true, y_pred)
    f1c = f1_score(y_true, y_pred, average=None, labels=[0,1])
    p0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    r0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    mac = f1c.mean()
    print(f"{name:45s} acc={acc:.4f} macroF1={mac:.4f} F1_0={f1c[0]:.4f} F1_1={f1c[1]:.4f} P0={p0:.4f} R0={r0:.4f}")
    results.append(dict(name=name, acc=acc, macro_f1=mac, f1_0=f1c[0], f1_1=f1c[1], precision_0=p0, recall_0=r0))

def best_threshold(oof, y_true):
    best=(0.5,-1,None)
    for t in np.arange(0.05,0.96,0.025):
        p=(oof>=t).astype(int); mac=f1_score(y_true,p,average='macro')
        if mac>best[1]: best=(t,mac,p)
    return best

results = []
log("=== BASELINES ===")
report("baseline: always 1", y_B, np.ones(len(y_B),int), results)
report("baseline: hard rule (=always 1 here)", y_B, hard_pred, results)

log("=== B0: retrieval-only ===")
oof0 = cv_eval(retrieval_cols)
t0,_,p0 = best_threshold(oof0, y_B)
report(f"B0: retrieval-only (t={t0:.3f})", y_B, p0, results)

log("=== B1: + edit/conflict (entity/number/negation/action-antonym) ===")
cols1 = retrieval_cols + edit_cols
oof1 = cv_eval(cols1)
t1,_,p1 = best_threshold(oof1, y_B)
report(f"B1: + edit/conflict (t={t1:.3f})", y_B, p1, results)

log("=== B2: + body evidence (title vs own body) ===")
cols2 = cols1 + body_cols
oof2 = cv_eval(cols2)
t2,_,p2 = best_threshold(oof2, y_B)
report(f"B2: + body evidence (t={t2:.3f})", y_B, p2, results)

log("=== B3: + negative-prototype/cluster features ===")
cols3 = cols2 + proto_cols
oof3 = cv_eval(cols3)
t3,_,p3 = best_threshold(oof3, y_B)
report(f"B3: + negative-prototype (t={t3:.3f})", y_B, p3, results)

log("=== B4: class-weight sweep (best feature set = B3) ===")
best_cw = None
for w0 in [1.0,1.5,2.0,3.0,4.0,5.0]:
    cw = {0:w0, 1:1.0}
    oof_w = cv_eval(cols3, class_weight=cw)
    t,mac,p = best_threshold(oof_w, y_B)
    report(f"B4: class_weight 0:{w0}:1 (t={t:.3f})", y_B, p, results)
    if best_cw is None or mac > best_cw[0]: best_cw = (mac, w0, oof_w)

log("=== SUMMARY TABLE ===")
rdf = pd.DataFrame(results)
print(rdf.round(4).to_string(index=False))
rdf.to_csv('bnopos_results.csv', index=False)
log(f"total runtime: {(time.time()-T0)/60:.1f} min")
