"""V14: from-scratch hierarchical BiGRU + headline-guided attention + pairwise interaction.
No pretrained embeddings/weights anywhere (compliant). Trained on ALL train rows via
StratifiedGroupKFold(group=content_hash), evaluated on the cold-start subset, compared
directly to v12's true-baseline (macroF1=0.6422, F1_0=0.3241).

Body is split into ~50-word chunks (corpus has no punctuation -- established earlier) which
serve as the "sentence" units for the hierarchical encoder and headline-guided attention.
"""
import os, sys, re, time, hashlib, unicodedata, string as _s
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score
import torch, torch.nn as nn, torch.nn.functional as F

SEED=42
torch.manual_seed(SEED); np.random.seed(SEED)
T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f"device={DEVICE}")

train = pd.read_csv(r"D:\Lomba\IFEST2026_DAC\data\train.csv")
def nh(x):
    x=unicodedata.normalize('NFKC',str(x)).lower(); return re.sub(r'\s+',' ',x).strip()
train['ch']=train['content'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
URL=re.compile(r'https?://\S+|www\.\S+'); HTML=re.compile(r'<[^>]+>'); WS=re.compile(r'\s+')
def clean(s):
    s=unicodedata.normalize('NFKC',str(s)); s=HTML.sub(' ',s); s=URL.sub(' ',s)
    return WS.sub(' ',s).strip()
train['T']=train['title'].apply(clean); train['C']=train['content'].apply(clean)
y=train['label'].values
log(f"train shape {train.shape}")

# ---- vocab (word-level, from-scratch, no pretrained) ----
def wtoks(s): return s.lower().split()
freq=Counter()
for t in train['T']: freq.update(wtoks(t))
for t in train['C']: freq.update(wtoks(t))
VOCAB_SIZE=25000
vocab = {'<pad>':0, '<unk>':1}
for w,_ in freq.most_common(VOCAB_SIZE-2):
    vocab[w]=len(vocab)
log(f"vocab size {len(vocab)}")

def encode(tok_list, maxlen):
    ids=[vocab.get(w,1) for w in tok_list[:maxlen]]
    if len(ids)<maxlen: ids += [0]*(maxlen-len(ids))
    return ids

MAXLEN_TITLE=24
MAXLEN_CHUNK=50
MAX_CHUNKS=10

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

log("pre-tokenizing (title ids, chunk ids, chunk mask)...")
title_ids = np.zeros((len(train), MAXLEN_TITLE), dtype=np.int64)
chunk_ids = np.zeros((len(train), MAX_CHUNKS, MAXLEN_CHUNK), dtype=np.int64)
chunk_mask = np.zeros((len(train), MAX_CHUNKS), dtype=np.float32)
for i,(T,C) in enumerate(zip(train['T'], train['C'])):
    title_ids[i] = encode(wtoks(T), MAXLEN_TITLE)
    cs = chunks_of(C)[:MAX_CHUNKS]
    for j,c in enumerate(cs):
        chunk_ids[i,j] = encode(wtoks(c), MAXLEN_CHUNK)
        chunk_mask[i,j] = 1.0
log("tokenizing done")

class HeadlineBodyDataset(torch.utils.data.Dataset):
    def __init__(self, idx):
        self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return (torch.from_numpy(title_ids[j]), torch.from_numpy(chunk_ids[j]),
                torch.from_numpy(chunk_mask[j]), torch.tensor(y[j], dtype=torch.float32))

class V14Model(nn.Module):
    def __init__(self, vocab_size, emb_dim=96, hid=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.title_gru = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.chunk_gru = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.doc_gru = nn.GRU(hid*2, hid, batch_first=True, bidirectional=True)
        self.attn_w = nn.Linear(hid*2, hid*2, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hid*2*4, hid*2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hid*2, 1)
        )
    def encode_title(self, title_ids):
        e = self.emb(title_ids)                      # B,T,E
        out,_ = self.title_gru(e)                     # B,T,2H
        mask = (title_ids!=0).float().unsqueeze(-1)
        H = (out*mask).sum(1)/mask.sum(1).clamp(min=1)  # mean-pool -> B,2H
        return H
    def encode_chunks(self, chunk_ids, chunk_mask):
        B,C,L = chunk_ids.shape
        flat = chunk_ids.view(B*C, L)
        e = self.emb(flat)
        out,_ = self.chunk_gru(e)                      # B*C,L,2H
        wmask = (flat!=0).float().unsqueeze(-1)
        S = (out*wmask).sum(1)/wmask.sum(1).clamp(min=1)  # B*C,2H
        S = S.view(B,C,-1)
        S = S * chunk_mask.unsqueeze(-1)                # zero-out empty chunks
        doc_out,_ = self.doc_gru(S)                     # hierarchical: B,C,2H
        return doc_out
    def forward(self, title_ids, chunk_ids, chunk_mask):
        H = self.encode_title(title_ids)                # B,2H
        S = self.encode_chunks(chunk_ids, chunk_mask)    # B,C,2H
        scores = torch.bmm(self.attn_w(S), H.unsqueeze(-1)).squeeze(-1)  # B,C
        scores = scores.masked_fill(chunk_mask==0, -1e9)
        a = F.softmax(scores, dim=1)                     # B,C
        Bh = (a.unsqueeze(-1)*S).sum(1)                   # headline-attended body -> B,2H
        pair = torch.cat([H, Bh, (H-Bh).abs(), H*Bh], dim=1)
        logit = self.mlp(pair).squeeze(-1)
        return logit

def train_fold(tr_idx, va_idx, epochs=6, batch_size=128, lr=1e-3, class_weight_0=6.0):
    model = V14Model(len(vocab)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_ds = HeadlineBodyDataset(tr_idx); va_ds = HeadlineBodyDataset(va_idx)
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=256, shuffle=False)
    pos_weight = torch.tensor([1.0]).to(DEVICE)  # weight for class 1 in BCEWithLogits (label=1 is majority)
    # we want higher loss weight for class 0 (minority, y=0) -> use per-sample weight instead
    for ep in range(epochs):
        model.train(); tot_loss=0.0; n=0
        for tb, cb, mb, yb in tr_loader:
            tb,cb,mb,yb = tb.to(DEVICE),cb.to(DEVICE),mb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad()
            logit = model(tb,cb,mb)
            w = torch.where(yb==0, torch.tensor(class_weight_0,device=DEVICE), torch.tensor(1.0,device=DEVICE))
            loss = F.binary_cross_entropy_with_logits(logit, yb, weight=w)
            loss.backward(); opt.step()
            tot_loss += loss.item()*len(yb); n+=len(yb)
        log(f"    epoch {ep+1}/{epochs} loss={tot_loss/n:.4f}")
    model.eval(); preds=np.zeros(len(va_idx))
    with torch.no_grad():
        off=0
        for tb, cb, mb, yb in va_loader:
            tb,cb,mb = tb.to(DEVICE),cb.to(DEVICE),mb.to(DEVICE)
            logit = model(tb,cb,mb)
            p = torch.sigmoid(logit).cpu().numpy()
            preds[off:off+len(p)] = p; off+=len(p)
    return preds

if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='tiny dry run: 1 fold, subsample, 2 epochs')
    args = ap.parse_args()

    groups = train['ch'].values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(train, y, groups))

    _cnt=train['ch'].value_counts()
    cold_mask = train['ch'].map(_cnt).eq(1).values

    if args.quick:
        log("QUICK DRY RUN: fold0 only, epochs=2")
        a,b = folds[0]
        rs = np.random.RandomState(0)
        a_small = rs.choice(a, size=min(2000,len(a)), replace=False)
        b_small = rs.choice(b, size=min(500,len(b)), replace=False)
        preds = train_fold(a_small, b_small, epochs=2, batch_size=64)
        yb = y[b_small]
        best=(0.5,-1)
        for t in np.arange(0.1,0.91,0.05):
            mac=f1_score(yb,(preds>=t).astype(int),average='macro')
            if mac>best[1]: best=(t,mac)
        print(f"quick dry-run fold0 subsample: macroF1={best[1]:.4f} @ t={best[0]:.2f}")
        print("(subsample is tiny, this is a SANITY CHECK only, not a real result)")
    else:
        oof = np.zeros(len(train))
        for i,(a,b) in enumerate(folds):
            log(f"=== fold {i} ===")
            preds = train_fold(a, b, epochs=6, batch_size=128)
            oof[b] = preds
            fold_cold = cold_mask[b]
            if fold_cold.sum()>0:
                yb=y[b][fold_cold]; pb=preds[fold_cold]
                mac=f1_score(yb,(pb>=0.5).astype(int),average='macro')
                log(f"  fold{i} cold macroF1@0.5={mac:.4f}")
        np.save(os.path.join(r"D:\Lomba\IFEST2026_DAC\notebook_v12\_diag_cache",'oof_v14.npy'), oof)
        cold_idx=np.where(cold_mask)[0]
        co=oof[cold_idx]; cy=y[cold_idx]
        best=(0.5,-1)
        for t in np.arange(0.05,0.96,0.025):
            mac=f1_score(cy,(co>=t).astype(int),average='macro')
            if mac>best[1]: best=(t,mac)
        pred=(co>=best[0]).astype(int)
        f1c=f1_score(cy,pred,average=None,labels=[0,1])
        p0=precision_score(cy,pred,pos_label=0,zero_division=0); r0=recall_score(cy,pred,pos_label=0,zero_division=0)
        print(f"\n=== V14 FINAL (cold subset, n={len(cold_idx)}) ===")
        print(f"threshold={best[0]:.3f} macroF1={best[1]:.4f} F1_0={f1c[0]:.4f} F1_1={f1c[1]:.4f} P0={p0:.4f} R0={r0:.4f}")
        print(f"reference: v12 true baseline macroF1=0.6422 F1_0=0.3241")
        log(f"total runtime: {(time.time()-T0)/60:.1f} min")
