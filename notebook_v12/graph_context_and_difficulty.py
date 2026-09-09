"""Two new, genuinely-untested compliant experiments for A_cold:
(1) Graph context features: entity co-occurrence graph built from the corpus (from-scratch,
    no pretrained anything). 1-hop = local window match for entities present in body;
    2-hop = does body contain a global co-occurrence neighbor of a MISSING title entity.
(2) Difficulty-aware majority (class-1) undersampling: use existing leak-free OOF
    confidence to keep hard class-1 training rows and drop easy ones, evaluated on the
    untouched validation fold.
"""
import os, sys, time, hashlib, re, unicodedata, string as _s
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedGroupKFold
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

# ---- entity pool (same as v12) ----
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
log(f"entity pool: {len(ENT)}")

# ---- global co-occurrence graph (entity -> Counter(neighbor words)), window +/-5 ----
log("building entity co-occurrence graph from corpus...")
WIN=5
neighbors = defaultdict(Counter)
for body in bodies:
    words = [w.strip(_s.punctuation) for w in body.split()]
    ent_positions = [(i,w) for i,w in enumerate(words) if w in ENT]
    if not ent_positions: continue
    n=len(words)
    for i,w in ent_positions:
        ctx = words[max(0,i-WIN):i] + words[i+1:i+WIN+1]
        neighbors[w].update(c for c in ctx if c.isalpha() and len(c)>2)
TOPK=15
ent_graph = {w: set([x for x,_ in cnt.most_common(TOPK)]) for w,cnt in neighbors.items()}
log(f"co-occurrence graph built for {len(ent_graph)} entities")

def graph_context_feats(title, body):
    Ttok = title.split()
    body_words_raw = [w.strip(_s.punctuation) for w in body.split()]
    body_set = set(body_words_raw)
    tents = [w for w in Ttok if w in ENT]
    if not tents:
        return dict(hop1_local_match=0.0, hop1_n_checked=0.0, hop2_indirect_support=0.0, hop2_n_missing=0.0)
    n = len(body_words_raw)
    title_rest = set(w for w in Ttok if w not in ENT and w.lower() not in {'dan','di','ke','yang','dari','untuk'})
    hop1_scores=[]; hop2_hits=0; n_missing=0
    for e in tents:
        if e in body_set:
            # 1-hop: local window around occurrence(s) vs rest of title
            positions=[i for i,w in enumerate(body_words_raw) if w==e]
            best_overlap=0.0
            for p in positions:
                window=set(body_words_raw[max(0,p-WIN):p]+body_words_raw[p+1:p+WIN+1])
                ov = len(window & title_rest)/max(len(title_rest),1)
                best_overlap=max(best_overlap,ov)
            hop1_scores.append(best_overlap)
        else:
            # 2-hop: entity missing -- does body contain a known graph-neighbor of it?
            n_missing += 1
            nbrs = ent_graph.get(e, set())
            if nbrs & body_set: hop2_hits += 1
    hop1 = float(np.mean(hop1_scores)) if hop1_scores else 0.0
    hop2 = hop2_hits/max(n_missing,1) if n_missing>0 else 0.0
    return dict(hop1_local_match=hop1, hop1_n_checked=float(len(hop1_scores)),
                hop2_indirect_support=hop2, hop2_n_missing=float(n_missing))

log("computing graph-context features for all rows...")
gc_rows = [graph_context_feats(t,b) for t,b in zip(titles,bodies)]
GCdf = pd.DataFrame(gc_rows)
print(GCdf.describe().round(4))

Ftr_gc = pd.concat([Ftr[FEAT48], GCdf], axis=1)

def run_cv(Xdf, model_fn):
    oof=np.zeros(len(train)); Xv=Xdf.values
    for a,b in folds:
        m=model_fn(); m.fit(Xv[a],y[a]); oof[b]=m.predict_proba(Xv[b])[:,1]
    return oof

def make_cat(): return cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                                              class_weights=[4,1], random_seed=SEED, verbose=False)

def best_macro(p, yy=None):
    yy = cy if yy is None else yy
    best=(0.5,-1)
    for t in np.arange(0.05,0.96,0.025):
        mac=f1_score(yy,(p>=t).astype(int),average='macro')
        if mac>best[1]: best=(t,mac)
    return best

def report(oof, name):
    co=oof[cold_idx]; t,mac=best_macro(co)
    pred=(co>=t).astype(int); f1c=f1_score(cy,pred,average=None,labels=[0,1])
    p0=precision_score(cy,pred,pos_label=0,zero_division=0); r0=recall_score(cy,pred,pos_label=0,zero_division=0)
    print(f"{name:45s} macroF1={mac:.4f} F1_0={f1c[0]:.4f} F1_1={f1c[1]:.4f} P0={p0:.4f} R0={r0:.4f}")
    return mac

log("=== Exp 1: 48-feat CatBoost(w=4) baseline vs + graph-context ===")
oof_base = run_cv(Ftr[FEAT48], make_cat); report(oof_base, "48-feat CatBoost w=4:1 (baseline)")
oof_gc = run_cv(Ftr_gc, make_cat); report(oof_gc, "+ graph-context (4 feats)")

log("=== Exp 2: difficulty-aware majority undersampling (uses oof_base as difficulty score) ===")
# difficulty for class-1 rows = OOF probability (lower = harder, model unsure it's class1)
for keep_frac in [1.0, 0.75, 0.5, 0.3]:
    oof_diff = np.zeros(len(train))
    for a,b in folds:
        ya = y[a]
        pos_a = a[ya==1]; neg_a = a[ya==0]
        pos_conf = oof_base[pos_a]   # OOF prob for these class-1 train rows (leak-free: computed by models never trained on them)
        order = np.argsort(-pos_conf)   # hardest (lowest conf) last when sorted descending by conf... sort ascending conf = hardest first
        order = np.argsort(pos_conf)    # ascending: hardest (lowest prob) first
        n_keep = int(len(pos_a)*keep_frac)
        keep_pos = pos_a[order[:n_keep]]   # keep the hardest keep_frac fraction
        a_filtered = np.concatenate([neg_a, keep_pos])
        m = make_cat(); m.fit(Ftr[FEAT48].values[a_filtered], y[a_filtered])
        oof_diff[b] = m.predict_proba(Ftr[FEAT48].values[b])[:,1]
    report(oof_diff, f"difficulty-aware keep_frac={keep_frac}")

log("=== Exp 3: combine graph-context + difficulty-aware (best keep_frac) ===")
# find best keep_frac from Exp2 quickly by rerunning search inline is done above; use 0.5 as a reasonable default then refine if needed
