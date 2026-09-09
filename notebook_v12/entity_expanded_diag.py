"""Expand the entity pool permissively (drop the title_df<0.02 cap that was excluding
recurring country/city/institution names like 'Amerika Serikat', 'Korea Selatan') and
re-measure entity-substitution / entity-role-conflict on the 320 A_cold false negatives.
Reuses cached oof/features/train from diagnostic_123.py.
"""
import os, sys, re, math, string as _s
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score

CACHE = r"D:\Lomba\IFEST2026_DAC\notebook_v12\_diag_cache"
oof = np.load(os.path.join(CACHE,'oof.npy'))
cold_mask = np.load(os.path.join(CACHE,'cold_mask.npy'))
train = pd.read_csv(os.path.join(CACHE,'train_slim.csv'))
y = train['label'].values

import unicodedata
URL=re.compile(r'https?://\S+|www\.\S+'); HTML=re.compile(r'<[^>]+>'); WS=re.compile(r'\s+')
def clean(s):
    s=unicodedata.normalize('NFKC',str(s)); s=HTML.sub(' ',s); s=URL.sub(' ',s)
    return WS.sub(' ',s).strip()
train['T']=train['title'].apply(clean); train['C']=train['content'].apply(clean)
titles=train['T'].tolist(); bodies=train['C'].tolist()

cold_idx = np.where(cold_mask)[0]
cold_oof = oof[cold_idx]; cold_y = y[cold_idx]
best=(0.5,-1)
for t in np.arange(0.05,0.96,0.025):
    mac=f1_score(cold_y,(cold_oof>=t).astype(int),average='macro')
    if mac>best[1]: best=(t,mac)
THR=best[0]
pred_cold=(cold_oof>=THR).astype(int)
FN_mask=(cold_y==0)&(pred_cold==1)
FN_idx=cold_idx[FN_mask]
print(f"threshold={THR:.3f}  FN={len(FN_idx)}")

# ---- expanded, permissive entity pool ----
up,lo=Counter(),Counter()
for t in train['C']:
    for w in t.split():
        c=w.strip(_s.punctuation)
        if not c.isalpha() or len(c)<3: continue
        if c[:1].isupper(): up[c]+=1
        else: lo[c.capitalize()]+=1
pur={w:up[w]/(up[w]+lo.get(w,0)) for w in up if up[w]>=5}          # min_count 8->5
tdf=Counter()
for t in train['T']:
    for w in set(t.split()): tdf[w]+=1
ENT_STRICT = {w for w,p in pur.items() if p>=0.95 and tdf.get(w,0)/len(train)<0.02}   # original
ENT_PERMISSIVE = {w for w,p in pur.items() if p>=0.85}                                # no title_df cap, lower purity
print(f"strict entity pool: {len(ENT_STRICT)}   permissive entity pool: {len(ENT_PERMISSIVE)}")
newly_added = ENT_PERMISSIVE - ENT_STRICT
print(f"newly recovered entities (sample 30): {sorted(list(newly_added))[:30]}")

STOP={'dan','di','ke','yang','dari','ini','itu','untuk','pada','dengan','akan','juga','saat'}

# ---- BM25 chunk index (cheap rebuild, no SVD needed) ----
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
chunk_txt=[chunks_of(c) for c in bodies]

def top_chunk(title, chunks):
    """cheapest relevant-chunk pick: highest word-overlap with title."""
    tw=set(toks(title))
    best_c, best_score = chunks[0], -1
    for c in chunks:
        cw=set(toks(c))
        score=len(tw&cw)
        if score>best_score: best_score=score; best_c=c
    return best_c

def entity_substitution_check(title, body, top_c, ENT):
    Ttok=title.split()
    tents=[w for w in Ttok if w in ENT]
    craw={w.strip(_s.punctuation) for w in body.split()}
    missing=[w for w in tents if w not in craw]
    if not missing:
        return False, None, None
    # same-context check: does the top-matching chunk contain an ALTERNATIVE entity
    # (from ENT pool, not equal to the missing title entity)?
    chunk_ents = [w.strip(_s.punctuation) for w in top_c.split() if w.strip(_s.punctuation) in ENT]
    for m in missing:
        alts = [e for e in chunk_ents if e != m]
        if alts:
            return True, m, alts[0]
    return False, missing[0], None

print("\n=== Q1: entity substitution / entity-role conflict share of FN, permissive pool ===")
results=[]
for idx in FN_idx:
    tc = top_chunk(titles[idx], chunk_txt[idx])
    fired, missing_ent, alt_ent = entity_substitution_check(titles[idx], bodies[idx], tc, ENT_PERMISSIVE)
    results.append(dict(idx=idx, fired=fired, missing_ent=missing_ent, alt_ent=alt_ent))
resdf = pd.DataFrame(results)
n_fired = resdf['fired'].sum()
print(f"entity-substitution / role-conflict (permissive pool + same-context check): {n_fired}/{len(FN_idx)} ({n_fired/len(FN_idx)*100:.1f}%)")

# also report simpler "any permissive entity missing" (without requiring an alternative present)
any_missing=0
for idx in FN_idx:
    Ttok=titles[idx].split()
    tents=[w for w in Ttok if w in ENT_PERMISSIVE]
    craw={w.strip(_s.punctuation) for w in bodies[idx].split()}
    if any(w not in craw for w in tents): any_missing+=1
print(f"(for reference) any permissive-entity missing from body, regardless of alternative: {any_missing}/{len(FN_idx)} ({any_missing/len(FN_idx)*100:.1f}%)")

print("\n=== Q2: 'same context + different entity' rule -- precision/recall/F1 class0 on ALL cold ===")
pred_rule = np.ones(len(cold_idx), dtype=int)  # default predict 1
fire_detail = {}
for i, idx in enumerate(cold_idx):
    tc = top_chunk(titles[idx], chunk_txt[idx])
    fired, missing_ent, alt_ent = entity_substitution_check(titles[idx], bodies[idx], tc, ENT_PERMISSIVE)
    if fired:
        pred_rule[i] = 0
        fire_detail[idx] = (missing_ent, alt_ent)
p0=precision_score(cold_y,pred_rule,pos_label=0,zero_division=0)
r0=recall_score(cold_y,pred_rule,pos_label=0,zero_division=0)
f10=f1_score(cold_y,pred_rule,pos_label=0,zero_division=0)
n_fire_total = (pred_rule==0).sum()
print(f"rule fires on {n_fire_total}/{len(cold_idx)} cold rows")
print(f"precision0={p0:.4f}  recall0={r0:.4f}  F1_0={f10:.4f}")
print(f"(compare: full 49-feature model F1_0=0.3252, best prior union-rule F1_0=0.184, entity_missing-only F1_0=0.256)")

print("\n=== Q3: 20 clearest subject/entity-substitution examples from top-100 most-confident FN ===")
fn_sorted = FN_idx[np.argsort(-oof[FN_idx])]
top100 = fn_sorted[:100]
clear_examples=[]
for idx in top100:
    tc = top_chunk(titles[idx], chunk_txt[idx])
    fired, missing_ent, alt_ent = entity_substitution_check(titles[idx], bodies[idx], tc, ENT_PERMISSIVE)
    if fired:
        clear_examples.append(dict(idx=idx, missing_ent=missing_ent, alt_ent=alt_ent, conf=oof[idx]))
print(f"({len(clear_examples)} of top-100 fired the entity-substitution rule)\n")
for i, ex in enumerate(clear_examples[:20]):
    idx=ex['idx']
    print(f"[{i+1}] title-entity MISSING='{ex['missing_ent']}'  body-alternative='{ex['alt_ent']}'  p(class1)={ex['conf']:.4f}")
    print(f"    title: {train['title'].iloc[idx][:100]}")
    print(f"    body : {train['content'].iloc[idx][:180]}...")
    print()
