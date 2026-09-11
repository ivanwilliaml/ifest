"""
E1: shared BiGRU-from-scratch pair encoder -> cosine(h,b) + L2 distance(h,b) as the ONLY
two new features, added on top of the existing 71 A_cold features. Fold-safe: vocab,
embedding and BiGRU are trained ONLY on each fold's train split (never touch fold-valid
rows), matching the same StratifiedGroupKFold(5, group=content_hash) used everywhere else.
Benchmark: 0.6995 (71 features, class_weights=[1,1]).
"""
import io, os, json, re, time
import numpy as np
import dill
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

d = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
torch.manual_seed(42)

with open(os.path.join(d, '_cache.dill'), 'rb') as f:
    ns = dill.load(f)
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
ns['Ftr'] = ns['_Ftr_base'].copy(); ns['Fte'] = ns['_Fte_base'].copy()
exec(compile(cells[7], os.path.join(d, '_fast_claim.py'), 'exec'), ns)
print(f"71 base features ready in {time.time()-t0:.1f}s")

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix
import catboost as cb

train = ns['train']; Ftr = ns['Ftr']; y = ns['y']; ALL_A_FEATS = ns['ALL_A_FEATS']; SEED = ns['SEED']
groups = train['ch'].values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(sgkf.split(train, y, groups))

TITLE = train['T'].astype(str).tolist()
BODY = train['C'].astype(str).tolist()

TOK = re.compile(r"[a-zA-Z0-9]+")
def tokenize(s): return TOK.findall(s.lower())

TITLE_TOK = [tokenize(t) for t in TITLE]
BODY_TOK = [tokenize(b) for b in BODY]

TITLE_MAXLEN = 30
BODY_MAXLEN = 250
EMB_DIM = 64
GRU_HID = 32          # bidirectional -> 64 raw, projected to 32
PROJ_DIM = 32
EPOCHS = 4
BATCH = 128
DEVICE = 'cpu'

def build_vocab(idxs, min_freq=2, max_vocab=15000):
    freq = {}
    for i in idxs:
        for tok in TITLE_TOK[i] + BODY_TOK[i]:
            freq[tok] = freq.get(tok, 0) + 1
    items = sorted(freq.items(), key=lambda x: -x[1])
    vocab = {'<pad>': 0, '<unk>': 1}
    for w, c in items:
        if c < min_freq: continue
        if len(vocab) >= max_vocab: break
        vocab[w] = len(vocab)
    return vocab

def encode(tokens, vocab, maxlen):
    ids = [vocab.get(w, 1) for w in tokens[:maxlen]]
    if len(ids) < maxlen: ids = ids + [0] * (maxlen - len(ids))
    return ids

class PairDataset(Dataset):
    def __init__(self, idxs, vocab, labels=None):
        self.idxs = idxs; self.vocab = vocab; self.labels = labels
    def __len__(self): return len(self.idxs)
    def __getitem__(self, k):
        i = self.idxs[k]
        t = encode(TITLE_TOK[i], self.vocab, TITLE_MAXLEN)
        b = encode(BODY_TOK[i], self.vocab, BODY_MAXLEN)
        lab = self.labels[i] if self.labels is not None else 0
        return torch.tensor(t), torch.tensor(b), torch.tensor(float(lab))

class SharedEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.gru = nn.GRU(EMB_DIM, GRU_HID, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(GRU_HID * 2, PROJ_DIM)
    def forward(self, x):
        e = self.emb(x)
        _, hn = self.gru(e)                     # hn: (2, B, GRU_HID)
        h = torch.cat([hn[0], hn[1]], dim=-1)    # (B, GRU_HID*2)
        return torch.tanh(self.proj(h))          # (B, PROJ_DIM)

class PairModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.enc = SharedEncoder(vocab_size)
        self.head = nn.Sequential(nn.Linear(PROJ_DIM * 4, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, t, b):
        h = self.enc(t); bb = self.enc(b)
        inter = torch.cat([h, bb, (h - bb).abs(), h * bb], dim=-1)
        return self.head(inter).squeeze(-1), h, bb

def cos_dist(h, b):
    hn = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
    bn = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    cos = (hn * bn).sum(-1)
    dist = (h - b).norm(dim=-1)
    return cos.detach().cpu().numpy(), dist.detach().cpu().numpy()

def macro_best(oof, cold_mask, y):
    cp = oof[cold_mask]; cy = y[cold_mask]
    best = (0.5, -1)
    for t in np.arange(0.05, 0.96, 0.025):
        mac = f1_score(cy, (cp >= t).astype(int), average='macro')
        if mac > best[1]: best = (t, mac)
    return best

_cnt = train['ch'].value_counts()
cold_mask = train['ch'].map(_cnt).eq(1).values

oof_bigru_cos = np.zeros(len(train))
oof_bigru_dist = np.zeros(len(train))
oof_cb = np.zeros(len(train))

for fi, (a, bidx) in enumerate(folds):
    tf0 = time.time()
    vocab = build_vocab(a)
    model = PairModel(len(vocab)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()

    ds_train = PairDataset(a, vocab, y)
    dl_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True)
    model.train()
    for ep in range(EPOCHS):
        tot = 0.0; n = 0
        for tt, bb_, lab in dl_train:
            tt, bb_, lab = tt.to(DEVICE), bb_.to(DEVICE), lab.to(DEVICE)
            opt.zero_grad()
            logit, _, _ = model(tt, bb_)
            loss = lossf(logit, lab)
            loss.backward(); opt.step()
            tot += loss.item() * len(lab); n += len(lab)
        # print(f"  fold{fi} epoch{ep} loss={tot/n:.4f}")

    model.eval()
    ds_valid = PairDataset(bidx, vocab, None)
    dl_valid = DataLoader(ds_valid, batch_size=256, shuffle=False)
    cos_all = []; dist_all = []
    with torch.no_grad():
        for tt, bb_, _ in dl_valid:
            tt, bb_ = tt.to(DEVICE), bb_.to(DEVICE)
            _, h, bbv = model(tt, bb_)
            c, dst = cos_dist(h, bbv)
            cos_all.append(c); dist_all.append(dst)
    oof_bigru_cos[bidx] = np.concatenate(cos_all)
    oof_bigru_dist[bidx] = np.concatenate(dist_all)

    Xa = Ftr[ALL_A_FEATS].values
    ds_a_train = PairDataset(a, vocab, None)
    dl_a_train = DataLoader(ds_a_train, batch_size=256, shuffle=False)
    cos_a = []; dist_a = []
    with torch.no_grad():
        for tt, bb_, _ in dl_a_train:
            tt, bb_ = tt.to(DEVICE), bb_.to(DEVICE)
            _, h, bbv = model(tt, bb_)
            c, dst = cos_dist(h, bbv)
            cos_a.append(c); dist_a.append(dst)
    cos_a = np.concatenate(cos_a); dist_a = np.concatenate(dist_a)

    Xa_train_ext = np.column_stack([Xa[a], cos_a, dist_a])
    Xa_valid_ext = np.column_stack([Xa[bidx], oof_bigru_cos[bidx], oof_bigru_dist[bidx]])

    m = cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                               class_weights=[1, 1], random_seed=SEED, verbose=False)
    m.fit(Xa_train_ext, y[a])
    oof_cb[bidx] = m.predict_proba(Xa_valid_ext)[:, 1]
    fold_thr, fold_mac = macro_best(oof_cb, cold_mask & np.isin(np.arange(len(train)), bidx), y)
    print(f"fold {fi}: vocab={len(vocab)} done in {time.time()-tf0:.1f}s  fold-only Macro F1={fold_mac:.4f} @ thr={fold_thr:.3f}", flush=True)

thr, mac = macro_best(oof_cb, cold_mask, y)
cold_pred = (oof_cb[cold_mask] >= thr).astype(int)
f1_0 = f1_score(y[cold_mask], cold_pred, pos_label=0)
f1_1 = f1_score(y[cold_mask], cold_pred, pos_label=1)
print(f"\nE1 (71 feats + bigru_cosine + bigru_dist): OOF Macro F1 = {mac:.4f} @ thr={thr:.3f}")
print(f"OOF F1-0 = {f1_0:.4f}  OOF F1-1 = {f1_1:.4f}")
print(f"Delta vs 0.6995 baseline: {mac-0.6995:+.4f}")
print(f"TOTAL wall time: {time.time()-t0:.1f}s")
