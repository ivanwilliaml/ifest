import nbformat as nbf
import textwrap

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(src).strip()))

def code(src):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(src).strip()))

# ============================================================
md("""
# Penyisihan IFEST 2026 DAC — V7: Rule Layer + Cold-Start Consistency Model

## The finding that reframes the task

Splitting validation by *generalization regime* (as the V7 spec asked for) exposed
something no previous version could see, because the grouped split used by V4-V6 contains
**zero** rows of the critical regime by construction:

| regime | share of val | class-0 rate |
|---|---|---|
| A cold-start (body unseen) | 67.4% | **6.0%** |
| B same content, new title | 10.0% | **60.4%** |
| C exact (title,content) pair seen | 22.6% | known from train |

Conditioning regime B further makes it near-deterministic:

```
P(label=0 | body seen in train, train already has a label=1 title for it, this title is new)
    = 0.977   (n=131)
P(label=0 | body seen in train, train has NO positive title yet)
    = 0.035   (n=86)
```

This follows directly from how the data was built: each article is paired with **one true
headline** (label 1) plus tampered variants (label 0). Once the true headline of article
C is visible in train, any *other* title on C is almost certainly a tampered one.

**Two deterministic rules alone — with no model at all, everything else predicted 1 —
score Macro F1 = 0.8539 on validation**, versus 0.75591 for the best trained system (V4).

### Test-set coverage of the rules

| regime | test rows | handling |
|---|---|---|
| C exact pair | 927 (25.7%) | lookup (0 label conflicts in train, so exact) |
| B, train has a positive | 216 (6.0%) | rule -> 0 (precision 0.977) |
| B, train has no positive | 167 (4.6%) | rule -> 1 (precision 0.965) |
| A cold start | 2293 (63.6%) | **model** |

36.4% of test is decided deterministically, and rule B alone should recover ~211 of the
~360 expected test negatives. V2-V6 let the (class-1-biased) model answer those 216 rows
and almost certainly got most of them wrong.

## Consequences for the model

The model's real job is now only the **2293 cold-start rows**, where class-0 is ~6%. That
retroactively justifies `GroupShuffleSplit` for *training* — the earlier mistake was
applying that model to rows the rules should have owned, not the split itself.

Because the rules already capture the negative-dense regimes, model capacity matters less
than it did. So V7 keeps the pipeline lean and spends its one experiment on the highest-value
open question from the literature review: **the encoder**.

## Encoder pilot (the one controlled variable)

Per the spec's "do not become another uncontrolled collection of variations": the pilot
changes **only** the encoder, holding the V4-proven recipe fixed.

- `indobenchmark/indobert-base-p1` — current, plain MLM, no entailment knowledge
- `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` — 279M params, pretrained on
  2.7M premise/hypothesis pairs across 27 languages (Indonesian included). The task *is*
  entailment, so this starts with the concept V4 had to learn from 1,437 real negatives.

Deliberately **not** implemented, with reasons:
- **Synthetic negatives** — V6 measured them at 0.735 vs V4's 0.756.
- **Cross-article mining** — V5 measured 0.744.
- **3-way entailment/contradiction/unsupported auxiliary head** — no reliable way to derive
  those labels from a binary target, so the head would train on almost nothing. The NLI
  checkpoint supplies that knowledge instead, pretrained rather than fabricated.
""")

# ============================================================
md("## 1. Environment & Config")

code("""
import os, sys, subprocess
def _pip(p):
    try: subprocess.run([sys.executable,'-m','pip','install','-q']+p, check=True, timeout=240)
    except Exception as e: print(f"[warn] pip {p}: {e}")
_pip(['-U','transformers','accelerate'])
_pip(['sentencepiece','protobuf'])
""")

code("""
import re, time, json, hashlib, unicodedata, warnings, random, string as _string, shutil
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import transformers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| CUDA", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))

TIMINGS={}
class Timer:
    def __init__(s,n): s.n=n
    def __enter__(s): s.t=time.time(); return s
    def __exit__(s,*a):
        d=time.time()-s.t; TIMINGS[s.n]=TIMINGS.get(s.n,0)+d; print(f"[TIMER] {s.n}: {d:.1f}s")
RUN_T0=time.time()
def rm_ckpt(p): shutil.rmtree(p, ignore_errors=True)

CONFIG = {
    'retriever_model':'intfloat/multilingual-e5-small',
    'chunk_words':50,'chunk_stride':25,'top_k_semantic':3,'top_k_lexical':2,
    'include_first_chunk':True,'retriever_max_len':96,'retriever_batch_size':256,
    'lexical_max_features':60000,
    'candidates':{'indobert':'indobenchmark/indobert-base-p1',
                  'mdeberta_nli':'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'},
    'max_len':384,'batch_size':16,'lr':2e-5,'epochs':3,'early_stopping_patience':1,
    'val_size':0.15,
    'pilot_rows':6000,'pilot_epochs':1,      # encoder pilot gate
    'contrastive_epochs':2,'contrastive_lr':1e-5,'contrastive_margin':0.15,'contrastive_batch_size':8,
    'threshold_grid':[round(x,2) for x in np.arange(0.10,0.91,0.05)],
}
print(json.dumps(CONFIG,indent=2,default=str))
""")

# ============================================================
md("## 2. Load data, hashes, and the rule layer's lookup structures")

code("""
with Timer("load"):
    F={}
    for r,d,fs in os.walk('/kaggle/input'):
        for f in fs:
            if f in ('train.csv','test.csv','sample_submission.csv'): F.setdefault(f,os.path.join(r,f))
    assert len(F)==3, F
    train=pd.read_csv(F['train.csv']); test=pd.read_csv(F['test.csv']); sample_sub=pd.read_csv(F['sample_submission.csv'])

def nh(s):
    s=unicodedata.normalize('NFKC',str(s)).lower(); return re.sub(r'\\s+',' ',s).strip()
for d in (train,test):
    d['ch']=d['content'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
    d['th']=d['title'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
print("train",train.shape,"test",test.shape)

def build_lookups(df):
    \"\"\"exact (title,content)->label, and content -> list of (title_hash,label) seen.\"\"\"
    pair_map = dict(zip(zip(df['th'],df['ch']), df['label']))
    by_content = defaultdict(list)
    for th,ch,lb in zip(df['th'],df['ch'],df['label']): by_content[ch].append((th,lb))
    return pair_map, by_content

def apply_rules(df, pair_map, by_content):
    \"\"\"Returns (regime, rule_label) per row. rule_label is None where no rule fires.\"\"\"
    regimes, labels = [], []
    for th,ch in zip(df['th'],df['ch']):
        if (th,ch) in pair_map:
            regimes.append('C_exact'); labels.append(pair_map[(th,ch)])
        elif ch in by_content:
            if any(l==1 for _,l in by_content[ch]):
                regimes.append('B_has_pos'); labels.append(0)   # 0.977 precise
            else:
                regimes.append('B_no_pos');  labels.append(1)   # 0.965 precise
        else:
            regimes.append('A_cold'); labels.append(None)
    return np.array(regimes), labels
""")

# ============================================================
md("""
## 3. Validate the rule layer on a test-mirroring split

The grouped split cannot contain regime B at all, so rule quality is verified on a
**random** split, whose regime composition (67/10/23) closely matches test (64/11/26).
""")

code("""
from sklearn.metrics import f1_score, confusion_matrix, classification_report

with Timer("rule_validation"):
    rng=np.random.RandomState(SEED); perm=rng.permutation(len(train)); cut=int(0.85*len(train))
    rtr, rva = perm[:cut], perm[cut:]
    pm, bc = build_lookups(train.iloc[rtr])
    reg, rule_lab = apply_rules(train.iloc[rva], pm, bc)
    y_rva = train['label'].iloc[rva].values

    print("regime composition of the random-split validation:")
    for r in ['A_cold','B_has_pos','B_no_pos','C_exact']:
        m = reg==r
        if m.sum():
            print(f"  {r:10s} n={m.sum():5d} ({m.mean()*100:5.1f}%)  class0={np.mean(y_rva[m]==0):.3f}")

    # precision of each rule where it fires
    print("\\nrule precision:")
    for r in ['C_exact','B_has_pos','B_no_pos']:
        m = reg==r
        if m.sum():
            pred_r = np.array([rule_lab[i] for i in np.where(m)[0]])
            print(f"  {r:10s} accuracy={np.mean(pred_r==y_rva[m]):.4f}  (n={m.sum()})")

    # rules only, everything else -> 1
    rules_only = np.array([1 if l is None else l for l in rule_lab])
    RULES_ONLY_F1 = f1_score(y_rva, rules_only, average='macro')
    print(f"\\nRULES ONLY (no model, cold-start all predicted 1): Macro F1 = {RULES_ONLY_F1:.4f}")
    print(f"  F1 class0={f1_score(y_rva,rules_only,pos_label=0):.4f}  F1 class1={f1_score(y_rva,rules_only,pos_label=1):.4f}")
    print(confusion_matrix(y_rva, rules_only))
""")

# ============================================================
md("""
## 4. Cold-start model — training split

The model only ever sees rows the rules cannot decide, so it is trained and validated on
cold-start-style data via `GroupShuffleSplit` on the content hash (no body appears on both
sides). This is the right split *for the model's actual job*.
""")

code("""
from sklearn.model_selection import GroupShuffleSplit
with Timer("split"):
    gss=GroupShuffleSplit(n_splits=1,test_size=CONFIG['val_size'],random_state=SEED)
    tr_idx,va_idx=next(gss.split(train,train['label'],groups=train['ch']))
    assert not (set(train['ch'].iloc[tr_idx]) & set(train['ch'].iloc[va_idx]))
    print(f"model-train {len(tr_idx)} | model-val {len(va_idx)} (class0 {np.mean(train['label'].iloc[va_idx]==0):.3f})")
""")

# ============================================================
md("## 5. Cleaning, entity mining, hybrid retrieval (unchanged from V4's proven recipe)")

code("""
URL_RE=re.compile(r'https?://\\S+|www\\.\\S+'); HTML_RE=re.compile(r'<[^>]+>'); WS_RE=re.compile(r'\\s+')
def clean(s):
    s=unicodedata.normalize('NFKC',str(s)); s=HTML_RE.sub(' ',s); s=URL_RE.sub(' ',s)
    return WS_RE.sub(' ',s).strip()
with Timer("clean"):
    for d in (train,test):
        d['title_clean']=d['title'].apply(clean); d['content_clean']=d['content'].apply(clean)

NUM_RE=re.compile(r'\\d+[.,]?\\d*\\s*%?')
with Timer("entity_mining"):
    up,lo=Counter(),Counter()
    for t in train['content_clean']:
        for w in t.split():
            c=w.strip(_string.punctuation)
            if not c.isalpha() or len(c)<3: continue
            if c[:1].isupper(): up[c]+=1
            else: lo[c.capitalize()]+=1
    purity={w:up[w]/(up[w]+lo.get(w,0)) for w in up if up[w]>=8}
    tdf=Counter()
    for t in train['title_clean']:
        for w in set(t.split()): tdf[w]+=1
    ENT={w for w,p in purity.items() if p>=0.95 and tdf.get(w,0)/len(train)<0.02}
    print("entity pool:",len(ENT))

def body_ents(s):
    out=set()
    for i,t in enumerate(s.split()):
        c=t.strip(_string.punctuation)
        if i>0 and c[:1].isupper() and c.isalpha() and len(c)>2: out.add(c.lower())
    return out

def conflict_text(title, content):
    craw={w.strip(_string.punctuation) for w in content.split()}
    parts=[]
    miss=[w for w in title.split() if w in ENT and w not in craw][:5]
    if miss: parts.append('ENTITAS JUDUL TAK ADA DI ISI: '+', '.join(miss))
    tn,cn=set(NUM_RE.findall(title)),set(NUM_RE.findall(content))
    nm=list(tn-cn)[:5]
    if nm: parts.append('ANGKA JUDUL TAK ADA DI ISI: '+', '.join(nm))
    be=[e for e in body_ents(content) if e not in title.lower()][:4]
    if be: parts.append('ENTITAS ISI: '+', '.join(be))
    return ' | '.join(parts)
""")

code("""
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer

def chunks_of(s,w=CONFIG['chunk_words'],st=CONFIG['chunk_stride']):
    ws=s.split()
    if not ws: return []
    if len(ws)<=w: return [' '.join(ws)]
    out=[]
    for i in range(0,len(ws),st):
        c=ws[i:i+w]
        if not c: break
        out.append(' '.join(c))
        if i+w>=len(ws): break
    return out

with Timer("retriever_load"):
    rtok=AutoTokenizer.from_pretrained(CONFIG['retriever_model'])
    rmod=AutoModel.from_pretrained(CONFIG['retriever_model']).to(DEVICE).eval()

def avg_pool(h,m):
    h=h.masked_fill(~m[...,None].bool(),0.0); return h.sum(1)/m.sum(1)[...,None]

@torch.no_grad()
def encode(texts, prefix, max_length=None):
    bs=CONFIG['retriever_batch_size']; ml=max_length or CONFIG['retriever_max_len']
    out=[]; amp=torch.cuda.is_available()
    for i in range(0,len(texts),bs):
        b=[prefix+t for t in texts[i:i+bs]]
        e=rtok(b,padding=True,truncation=True,max_length=ml,return_tensors='pt').to(DEVICE)
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp): o=rmod(**e)
        v=torch.nn.functional.normalize(avg_pool(o.last_hidden_state,e['attention_mask']),p=2,dim=1)
        out.append(v.float().cpu().numpy())
    return np.concatenate(out,0) if out else np.zeros((0,rmod.config.hidden_size))

with Timer("retrieval"):
    comb=pd.concat([train[['ch','th','content_clean','title_clean']],
                    test[['ch','th','content_clean','title_clean']]],ignore_index=True)
    uc=comb.drop_duplicates('ch')[['ch','content_clean']].reset_index(drop=True)
    flat,rng_map=[],{}
    for h,c in zip(uc['ch'],uc['content_clean']):
        cs=chunks_of(c); s=len(flat); flat.extend(cs if cs else ['']); rng_map[h]=(s,len(flat))
    print(f"unique articles {len(uc)} chunks {len(flat)}")
    cemb=encode(flat,'passage: ')
    ut=comb.drop_duplicates('th')[['th','title_clean']].reset_index(drop=True)
    temb=dict(zip(ut['th'],encode(ut['title_clean'].tolist(),'query: ',max_length=32)))
    lv=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=CONFIG['lexical_max_features'])
    lv.fit(pd.concat([ut['title_clean'],pd.Series(flat)]))
    ctf=lv.transform(flat); ttf=dict(zip(ut['th'],lv.transform(ut['title_clean'])))

    def retrieve(df):
        res=[]
        for h,th in zip(df['ch'],df['th']):
            s,e=rng_map[h]; ct=flat[s:e]; n=len(ct)
            sem=cemb[s:e]@temb[th]
            lex=np.asarray((ctf[s:e]@ttf[th].T).todense()).ravel()
            sel=set(np.argsort(-sem)[:min(CONFIG['top_k_semantic'],n)].tolist())|\\
                set(np.argsort(-lex)[:min(CONFIG['top_k_lexical'],n)].tolist())
            if CONFIG['include_first_chunk']: sel.add(0)
            res.append(' '.join(ct[i] for i in sorted(sel)))
        return res
    for d in (train,test):
        d['retrieved']=retrieve(d)
        d['body']=d['retrieved']+[(' [SEP] '+conflict_text(t,c)) if conflict_text(t,c) else ''
                                   for t,c in zip(d['title_clean'],d['content_clean'])]
    del cemb, ctf
""")

# ============================================================
md("## 6. Training machinery (V4 recipe: class-weighted CE + same-content pairwise phase)")

code("""
from transformers import (AutoModelForSequenceClassification, Trainer, TrainingArguments,
                           EarlyStoppingCallback)

class PairDS(Dataset):
    def __init__(s,t,b,y,tok,ml): s.t=list(t); s.b=list(b); s.y=None if y is None else list(y); s.tok=tok; s.ml=ml
    def __len__(s): return len(s.t)
    def __getitem__(s,i):
        e=s.tok(s.t[i],s.b[i],truncation=True,max_length=s.ml,padding='max_length',return_tensors='pt')
        it={k:v.squeeze(0) for k,v in e.items()}
        if s.y is not None: it['labels']=torch.tensor(s.y[i],dtype=torch.long)
        return it

def metrics(ep):
    lg,lb=ep; p=np.argmax(lg,axis=-1); return {'macro_f1':f1_score(lb,p,average='macro')}

class WTrainer(Trainer):
    def __init__(s,*a,class_weights=None,**k): super().__init__(*a,**k); s.cw=class_weights
    def compute_loss(s,model,inputs,return_outputs=False,**k):
        y=inputs.pop('labels'); o=model(**inputs)
        l=nn.CrossEntropyLoss(weight=s.cw.to(o.logits.device))(o.logits,y)
        return (l,o) if return_outputs else l

def cweights(y):
    n=len(y); n0=(y==0).sum(); n1=(y==1).sum(); return torch.tensor([n/(2*n0),n/(2*n1)],dtype=torch.float)

def make_trainer(model_name, tt,tb,ty, vt,vb,vy, epochs, out_dir, early=True, fp16=None, bs=None):
    ml=CONFIG['max_len']; bs=bs or CONFIG['batch_size']
    use_fp16 = torch.cuda.is_available() if fp16 is None else (fp16 and torch.cuda.is_available())
    tok=AutoTokenizer.from_pretrained(model_name)
    # NLI checkpoints ship a 3-way head; ignore_mismatched_sizes swaps in a fresh 2-way head
    # while keeping the entailment-shaped encoder body, which is the part we want.
    mdl=AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=2,ignore_mismatched_sizes=True)
    base=dict(output_dir=out_dir,num_train_epochs=epochs,per_device_train_batch_size=bs,
              learning_rate=CONFIG['lr'],weight_decay=0.01,fp16=use_fp16,dataloader_num_workers=0,
              seed=SEED,logging_steps=100,report_to=[],disable_tqdm=False)
    if early:
        args=TrainingArguments(per_device_eval_batch_size=bs*2,eval_strategy='epoch',save_strategy='epoch',
                               save_total_limit=1,load_best_model_at_end=True,
                               metric_for_best_model='macro_f1',greater_is_better=True,**base)
        t=WTrainer(model=mdl,args=args,train_dataset=PairDS(tt,tb,ty,tok,ml),
                   eval_dataset=PairDS(vt,vb,vy,tok,ml),compute_metrics=metrics,
                   class_weights=cweights(np.array(ty)),
                   callbacks=[EarlyStoppingCallback(early_stopping_patience=CONFIG['early_stopping_patience'])])
    else:
        args=TrainingArguments(eval_strategy='no',save_strategy='no',**base)
        t=WTrainer(model=mdl,args=args,train_dataset=PairDS(tt,tb,ty,tok,ml),class_weights=cweights(np.array(ty)))
    return t,tok

def probs_of(trainer, ds):
    return torch.softmax(torch.tensor(trainer.predict(ds).predictions),dim=-1).numpy()[:,1]
""")

# ============================================================
md("""
## 7. V7-A: encoder pilot gate

Only the encoder varies. Both candidates get the identical subset, epoch budget and
validation set. mDeBERTa is tried with fp16 first and falls back to fp32 automatically —
V3 hit `expected scalar type Half but found Float` on a DeBERTa-v3 variant.
""")

code("""
PILOT={}
with Timer("pilot"):
    sub_idx = tr_idx[:CONFIG['pilot_rows']]
    pt,pb,py = train['title_clean'].iloc[sub_idx], train['body'].iloc[sub_idx], train['label'].iloc[sub_idx].values
    vt,vb,vy = train['title_clean'].iloc[va_idx], train['body'].iloc[va_idx], train['label'].iloc[va_idx].values

    for name,mid in CONFIG['candidates'].items():
        for attempt,(fp16,bs) in enumerate([(True,CONFIG['batch_size']),(False,8)]):
            try:
                t0=time.time()
                tr_,tok_=make_trainer(mid,pt,pb,py,vt,vb,vy,CONFIG['pilot_epochs'],
                                       f'/kaggle/working/pilot_{name}',early=False,fp16=fp16,bs=bs)
                tr_.train()
                p=probs_of(tr_,PairDS(vt,vb,vy,tok_,CONFIG['max_len']))
                best=max(f1_score(vy,(p>=t).astype(int),average='macro') for t in CONFIG['threshold_grid'])
                PILOT[name]={'macro_f1':best,'seconds':time.time()-t0,'fp16':fp16}
                print(f"  [{name}] pilot Macro F1={best:.4f}  ({time.time()-t0:.0f}s, fp16={fp16})")
                rm_ckpt(f'/kaggle/working/pilot_{name}')
                del tr_; torch.cuda.empty_cache() if torch.cuda.is_available() else None
                break
            except Exception as e:
                print(f"  [{name}] attempt fp16={fp16} failed: {type(e).__name__}: {str(e)[:160]}")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                if attempt==1: PILOT[name]={'macro_f1':-1,'seconds':0,'fp16':None}

print("\\npilot results:", json.dumps(PILOT,indent=2,default=str))
BEST_ENCODER = CONFIG['candidates'][max(PILOT,key=lambda k:PILOT[k]['macro_f1'])]
BEST_FP16 = PILOT[max(PILOT,key=lambda k:PILOT[k]['macro_f1'])]['fp16']
print("selected encoder:",BEST_ENCODER,"(fp16=",BEST_FP16,")")
""")

# ============================================================
md("## 8. Full cold-start model + same-content pairwise phase")

code("""
with Timer("full_training"):
    t0=time.time()
    trainer,tok = make_trainer(BEST_ENCODER,
        train['title_clean'].iloc[tr_idx], train['body'].iloc[tr_idx], train['label'].iloc[tr_idx].values,
        vt,vb,vy, CONFIG['epochs'], '/kaggle/working/main', early=True, fp16=BEST_FP16)
    trainer.train()
    train_secs=time.time()-t0
val_p1 = probs_of(trainer, trainer.eval_dataset)
f1_p1 = max(f1_score(vy,(val_p1>=t).astype(int),average='macro') for t in CONFIG['threshold_grid'])
print(f"phase1 (classification) best-threshold Macro F1 on cold-start val: {f1_p1:.4f}")
rm_ckpt('/kaggle/working/main')
""")

code("""
# --- same-content pairwise phase (V4's only intervention that ever helped: +0.020) ---
def same_content_pairs(df, allowed):
    out=[]
    s=df[df['ch'].isin(allowed)]
    for h,g in s.groupby('ch'):
        pos=g[g.label==1]; neg=g[g.label==0]
        if len(pos)==0 or len(neg)==0: continue
        for _,p in pos.iterrows():
            for _,n in neg.iterrows():
                out.append({'tp':p['title_clean'],'bp':p['body'],'tn':n['title_clean'],'bn':n['body']})
    return out

class CPairDS(Dataset):
    def __init__(s,pairs,tok,ml): s.p=pairs; s.tok=tok; s.ml=ml
    def __len__(s): return len(s.p)
    def __getitem__(s,i):
        p=s.p[i]
        a=s.tok(p['tp'],p['bp'],truncation=True,max_length=s.ml,padding='max_length',return_tensors='pt')
        b=s.tok(p['tn'],p['bn'],truncation=True,max_length=s.ml,padding='max_length',return_tensors='pt')
        return {'pi':a['input_ids'].squeeze(0),'pm':a['attention_mask'].squeeze(0),
                'ni':b['input_ids'].squeeze(0),'nm':b['attention_mask'].squeeze(0)}

def contrastive(model,tok,pairs,label=''):
    if not pairs:
        print(f"[{label}] no pairs"); return model
    dl=DataLoader(CPairDS(pairs,tok,CONFIG['max_len']),batch_size=CONFIG['contrastive_batch_size'],
                  shuffle=True,num_workers=0)
    opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['contrastive_lr'])
    lf=nn.MarginRankingLoss(margin=CONFIG['contrastive_margin'])
    amp=torch.cuda.is_available() and bool(BEST_FP16)
    sc=torch.amp.GradScaler('cuda',enabled=amp)
    model.to(DEVICE).train()
    for ep in range(CONFIG['contrastive_epochs']):
        tot=0.0
        for b in dl:
            b={k:v.to(DEVICE) for k,v in b.items()}
            opt.zero_grad()
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):
                ps=torch.softmax(model(input_ids=b['pi'],attention_mask=b['pm']).logits.float(),-1)[:,1]
                ns=torch.softmax(model(input_ids=b['ni'],attention_mask=b['nm']).logits.float(),-1)[:,1]
                loss=lf(ps,ns,torch.ones_like(ps))
            sc.scale(loss).backward(); sc.step(opt); sc.update(); tot+=loss.item()
        print(f"[{label}] contrastive ep{ep+1} loss={tot/max(len(dl),1):.4f} ({len(pairs)} pairs)")
    model.eval(); return model

with Timer("contrastive_phase"):
    pairs_tr = same_content_pairs(train, set(train['ch'].iloc[tr_idx]))
    print(f"same-content pairs (train side): {len(pairs_tr)}")
    trainer.model = contrastive(trainer.model, tok, pairs_tr, 'val-run')
val_p2 = probs_of(trainer, trainer.eval_dataset)
f1_p2 = max(f1_score(vy,(val_p2>=t).astype(int),average='macro') for t in CONFIG['threshold_grid'])
print(f"phase1+2 best-threshold Macro F1 on cold-start val: {f1_p2:.4f}  (phase1 was {f1_p1:.4f})")
USE_P2 = f1_p2 >= f1_p1
val_probs = val_p2 if USE_P2 else val_p1
print("USE_PHASE2 =",USE_P2)
""")

# ============================================================
md("""
## 9. Threshold tuning in the presence of the rule layer

The threshold is chosen to maximise **global** Macro F1 on the test-mirroring random
split with rules applied first — not on cold-start alone, since rules already own most
negatives and that changes the optimal operating point.
""")

code("""
with Timer("threshold"):
    # A threshold tuned on cold-start alone is the wrong operating point: at inference the
    # rules already supply ~59% of all negatives, so the model should be pushed toward
    # RECALL on the few cold-start negatives. We therefore build a synthetic global
    # validation with the test regime mixture (63.6% cold / 25.7% C / 6.0% B_has_pos /
    # 4.6% B_no_pos) and optimise Macro F1 over the whole thing.
    TEST_MIX={'A_cold':0.636,'C_exact':0.257,'B_has_pos':0.060,'B_no_pos':0.046}
    rule_rows=[]   # (true_label, rule_prediction) taken from the random split, where rules fire
    for r,lab,truth in zip(reg, rule_lab, y_rva):
        if lab is not None: rule_rows.append((r,truth,lab))
    rule_df=pd.DataFrame(rule_rows,columns=['regime','y','pred'])

    n_cold=len(vy)
    total=int(n_cold/TEST_MIX['A_cold'])
    rs=np.random.RandomState(SEED)
    sampled=[]
    for r in ['C_exact','B_has_pos','B_no_pos']:
        pool=rule_df[rule_df.regime==r]
        k=int(round(TEST_MIX[r]*total))
        if len(pool)==0 or k==0: continue
        idx=rs.choice(len(pool),size=k,replace=len(pool)<k)
        sampled.append(pool.iloc[idx])
    rule_part=pd.concat(sampled,ignore_index=True) if sampled else pd.DataFrame(columns=['regime','y','pred'])
    print(f"synthetic global val: {n_cold} cold + {len(rule_part)} rule-decided = {n_cold+len(rule_part)}")

    rows=[]
    for t in CONFIG['threshold_grid']:
        cold_pred=(val_probs>=t).astype(int)
        y_all=np.concatenate([vy, rule_part['y'].values]) if len(rule_part) else vy
        p_all=np.concatenate([cold_pred, rule_part['pred'].values]) if len(rule_part) else cold_pred
        rows.append({'threshold':t,
                     'global_macro_f1':f1_score(y_all,p_all,average='macro'),
                     'cold_macro_f1':f1_score(vy,cold_pred,average='macro'),
                     'cold_recall0':((cold_pred==0)&(vy==0)).sum()/max((vy==0).sum(),1),
                     'cold_prec0':((cold_pred==0)&(vy==0)).sum()/max((cold_pred==0).sum(),1)})
    tdf=pd.DataFrame(rows); print(tdf.round(4).to_string(index=False))
    BEST_THRESHOLD=float(tdf.loc[tdf['global_macro_f1'].idxmax(),'threshold'])
    GLOBAL_F1=float(tdf['global_macro_f1'].max())
    COLD_F1=float(tdf.loc[tdf['global_macro_f1'].idxmax(),'cold_macro_f1'])
print(f"chosen threshold {BEST_THRESHOLD}: global Macro F1 {GLOBAL_F1:.4f} (cold-only {COLD_F1:.4f})")
print(f"rules-only reference (no model): {RULES_ONLY_F1:.4f}")
""")

# ============================================================
md("## 10. Final model on all data, then rules + model -> submission")

code("""
with Timer("final_train"):
    hist=[h for h in trainer.state.log_history if 'eval_macro_f1' in h]
    fe = max(1,int(round(max(hist,key=lambda h:h['eval_macro_f1']).get('epoch',CONFIG['epochs'])))) if hist else CONFIG['epochs']
    print("final epochs:",fe)
    ftr,ftok = make_trainer(BEST_ENCODER, train['title_clean'], train['body'], train['label'].values,
                             None,None,None, fe, '/kaggle/working/final', early=False, fp16=BEST_FP16)
    ftr.train()
    if USE_P2:
        ftr.model = contrastive(ftr.model, ftok, same_content_pairs(train,set(train['ch'])), 'final')
    rm_ckpt('/kaggle/working/final')

with Timer("inference"):
    tp = probs_of(ftr, PairDS(test['title_clean'],test['body'],None,ftok,CONFIG['max_len']))
    model_pred=(tp>=BEST_THRESHOLD).astype(int)

with Timer("submission"):
    pm_full, bc_full = build_lookups(train)          # rules use ALL of train at test time
    reg_test, rule_lab = apply_rules(test, pm_full, bc_full)
    final=np.array([model_pred[i] if rule_lab[i] is None else rule_lab[i] for i in range(len(test))])
    print("test regime coverage:")
    for r in ['A_cold','B_has_pos','B_no_pos','C_exact']:
        m=reg_test==r
        if m.sum(): print(f"  {r:10s} n={m.sum():5d} ({m.mean()*100:4.1f}%)  positives={final[m].mean():.3f}")
    changed=sum(1 for i in range(len(test)) if rule_lab[i] is not None and rule_lab[i]!=model_pred[i])
    print(f"rows where a rule overrode the model: {changed}")

    sub=pd.DataFrame({'id':test['id'],'label':final.astype(int)}).set_index('id').loc[test['id']].reset_index()
    assert sub.shape[0]==len(test) and set(sub['id'])==set(sample_sub['id'])
    assert set(sub['label'].unique())<={0,1} and list(sub.columns)==['id','label']
    sub.to_csv('/kaggle/working/submission.csv',index=False)
print(sub['label'].value_counts(normalize=True))
""")

code("""
T=time.time()-RUN_T0
print("="*60)
print("V7 — rule layer + cold-start model")
print(f"encoder selected      : {BEST_ENCODER}")
print(f"pilot                 : "+", ".join(f"{k}={v['macro_f1']:.4f}" for k,v in PILOT.items()))
print(f"rules-only Macro F1   : {RULES_ONLY_F1:.4f}   (random split, mirrors test composition)")
print(f"cold-start model F1   : {COLD_F1:.4f} @ threshold {BEST_THRESHOLD}  (phase2 used: {USE_P2})")
print(f"global Macro F1 (synth): {GLOBAL_F1:.4f}")
print(f"total runtime         : {T/60:.1f} min")
print("="*60)
for k,v in sorted(TIMINGS.items(),key=lambda kv:-kv[1]): print(f"  {k}: {v:.1f}s")
""")

nb['cells']=cells
nb['metadata']['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
nb['metadata']['language_info']={'name':'python','version':'3.11'}
with open('ifest2026_dac_v7.ipynb','w',encoding='utf-8') as f: nbf.write(nb,f)
print("Notebook written.")
