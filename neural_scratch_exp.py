"""
From-scratch neural experiment (NO pretrained anything).

Models
  A  classical control : TF-IDF(word+char) -> LogReg, fit on the SAME split
  B  Siamese CNN       : embedding -> multi-kernel Conv1d -> maxpool -> interaction
  C  Small Transformer : embedding+pos -> 2 layers, d=128, 4 heads -> mean-pool -> interaction

Interaction head is [h_t, h_c, |h_t-h_c|, h_t*h_c] -> MLP, as specified.

Subsampling is ASYMMETRIC on purpose: every negative is kept, positives are
downsampled. Negatives are the scarce class (1,437 total, 487 of them cold), so
cutting them would make a null result uninterpretable.

Evaluation is on COLD-START rows only (grouped split on normalized content),
because that is the only regime a model decides -- rules own the rest.

Usage:  python neural_scratch_exp.py [--pos N] [--epochs N] [--dim N] [--quick]
"""
import argparse, re, string, time, unicodedata, math
import numpy as np, pandas as pd, torch, torch.nn as nn
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument('--pos', type=int, default=4000, help='positives kept (all negatives always kept)')
ap.add_argument('--epochs', type=int, default=8)
ap.add_argument('--dim', type=int, default=128)
ap.add_argument('--vocab', type=int, default=20000)
ap.add_argument('--tlen', type=int, default=32)
ap.add_argument('--clen', type=int, default=256)
ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--quick', action='store_true', help='tiny smoke test')
A = ap.parse_args()
if A.quick:
    A.pos, A.epochs, A.dim, A.clen, A.vocab = 800, 2, 64, 128, 5000

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cuda':
    # torch.cuda.is_available() is NOT enough: the wheel may lack kernels for
    # this GPU's compute capability (Kaggle can hand out a Pascal P100, which
    # recent torch builds no longer compile for). Probe with a real op.
    try:
        print('gpu:', torch.cuda.get_device_name(0),
              'sm_%d%d' % torch.cuda.get_device_capability(0))
        _e = nn.Embedding(8, 8).to('cuda')
        _e(torch.zeros(2, 2, dtype=torch.long, device='cuda')).sum().backward()
        torch.cuda.synchronize()
    except Exception as _ex:
        print('CUDA unusable (%s: %s) -> falling back to CPU'
              % (type(_ex).__name__, str(_ex)[:120]))
        dev = 'cpu'
torch.set_num_threads(6)
print(f'device={dev}  pos={A.pos} epochs={A.epochs} dim={A.dim} clen={A.clen}')

# ------------------------------------------------------------------ load
D = r'D:\Lomba\IFEST2026_DAC\data'
import os
if not os.path.isdir(D):
    for r, _, fs in os.walk('/kaggle/input'):
        if 'train.csv' in fs: D = r; break
tr = pd.read_csv(os.path.join(D, 'train.csv'))
TC = 'title' if 'title' in tr.columns else tr.columns[1]
CC = 'content' if 'content' in tr.columns else tr.columns[2]
norm = lambda s: re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(s))).strip()
tr['T'] = tr[TC].map(norm); tr['C'] = tr[CC].map(norm)
tr['ch'] = tr['C'].str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True).map(hash)

# -------------------------------------------- grouped split (cold-start regime)
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
tr_i, va_i = next(gss.split(tr, tr.label, groups=tr['ch']))
TRA, VAL = tr.iloc[tr_i].copy(), tr.iloc[va_i].copy()
print(f'grouped split: train {len(TRA)} / val {len(VAL)}  val class0={1-VAL.label.mean():.4f}')

# ------------------------------------------ asymmetric subsample (train only)
neg = TRA[TRA.label == 0]
pos = TRA[TRA.label == 1].sample(n=min(A.pos, (TRA.label == 1).sum()), random_state=SEED)
SUB = pd.concat([neg, pos]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f'subsample: {len(SUB)} rows  neg={len(neg)} (ALL kept)  pos={len(pos)}  '
      f'ratio 1:{len(pos)/max(len(neg),1):.1f}')

# ---------------------------------------------- vocab from TRAIN text only
tokre = re.compile(r"[\w']+")
tok = lambda s: tokre.findall(s.lower())
cnt = Counter()
for s in pd.concat([SUB['T'], SUB['C']]):
    cnt.update(tok(s))
itos = ['<pad>', '<unk>'] + [w for w, _ in cnt.most_common(A.vocab - 2)]
stoi = {w: i for i, w in enumerate(itos)}
print(f'vocab built from train subsample only: {len(itos)}')


def enc(texts, L):
    out = np.zeros((len(texts), L), dtype=np.int64)
    for i, s in enumerate(texts):
        ids = [stoi.get(w, 1) for w in tok(s)[:L]]
        out[i, :len(ids)] = ids
    return torch.from_numpy(out)


Xt_tr, Xc_tr = enc(SUB['T'].tolist(), A.tlen), enc(SUB['C'].tolist(), A.clen)
Xt_va, Xc_va = enc(VAL['T'].tolist(), A.tlen), enc(VAL['C'].tolist(), A.clen)
y_tr = torch.tensor(SUB.label.values, dtype=torch.float32)
y_va = VAL.label.values


# ------------------------------------------------------------------ models
class Interact(nn.Module):
    def __init__(s, h):
        super().__init__()
        s.mlp = nn.Sequential(nn.Linear(4 * h, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(s, a, b):
        return s.mlp(torch.cat([a, b, (a - b).abs(), a * b], -1)).squeeze(-1)


class SiameseCNN(nn.Module):
    def __init__(s, V, d, nf=64, ks=(2, 3, 4, 5)):
        super().__init__()
        s.emb = nn.Embedding(V, d, padding_idx=0)
        s.convs = nn.ModuleList([nn.Conv1d(d, nf, k, padding=k // 2) for k in ks])
        s.h = nf * len(ks)
        s.head = Interact(s.h)

    def encode(s, x):
        e = s.emb(x).transpose(1, 2)
        return torch.cat([torch.relu(c(e)).max(-1).values for c in s.convs], -1)

    def forward(s, t, c):
        return s.head(s.encode(t), s.encode(c))


class SmallTransformer(nn.Module):
    def __init__(s, V, d, L=512, layers=2, heads=4):
        super().__init__()
        s.emb = nn.Embedding(V, d, padding_idx=0)
        s.pos = nn.Embedding(L, d)
        enc_l = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        s.tf = nn.TransformerEncoder(enc_l, layers)
        s.h = d
        s.head = Interact(d)

    def encode(s, x):
        m = x == 0
        p = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        z = s.tf(s.emb(x) + s.pos(p), src_key_padding_mask=m)
        z = z.masked_fill(m.unsqueeze(-1), 0.0)
        return z.sum(1) / (~m).sum(1, keepdim=True).clamp(min=1)

    def forward(s, t, c):
        return s.head(s.encode(t), s.encode(c))


def best_macro_f1(p, y):
    b = (0, 0.5)
    for t in np.arange(0.05, 0.96, 0.025):
        f = f1_score(y, (p >= t).astype(int), average='macro')
        if f > b[0]: b = (f, t)
    return b


def run(model, name):
    model = model.to(dev)
    n = sum(p.numel() for p in model.parameters())
    # label 1 is the majority, so up-weight the RARE class (0)
    w = torch.where(y_tr == 0, (y_tr == 1).sum() / max((y_tr == 0).sum(), 1), torch.tensor(1.0))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    idx = np.arange(len(y_tr))
    best = (0, 0, 0)
    t0 = time.time()
    for ep in range(A.epochs):
        model.train(); np.random.shuffle(idx); tot = 0.0
        for i in range(0, len(idx), A.bs):
            b = idx[i:i + A.bs]
            t, c, yy = Xt_tr[b].to(dev), Xc_tr[b].to(dev), y_tr[b].to(dev)
            ww = w[b].to(dev)
            logit = model(t, c)
            loss = (nn.functional.binary_cross_entropy_with_logits(
                logit, yy, reduction='none') * ww).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.detach().item() * len(b)
        model.eval(); ps = []
        with torch.no_grad():
            for i in range(0, len(y_va), 256):
                ps.append(torch.sigmoid(model(Xt_va[i:i + 256].to(dev),
                                              Xc_va[i:i + 256].to(dev))).cpu().numpy())
        p = np.concatenate(ps)
        f, th = best_macro_f1(p, y_va)
        ap_ = average_precision_score(1 - y_va, 1 - p)
        auc = roc_auc_score(1 - y_va, 1 - p)
        print(f'  {name} ep{ep+1} loss={tot/len(idx):.4f} valF1={f:.4f}@{th:.2f} '
              f'AP0={ap_:.4f} AUC0={auc:.4f}')
        if f > best[0]: best = (f, ap_, auc); bp = p
    print(f'  {name}: params={n/1e6:.2f}M  BEST valF1={best[0]:.4f} AP0={best[1]:.4f} '
          f'AUC0={best[2]:.4f}  [{time.time()-t0:.0f}s]')
    return best, bp


# --------------------------------------------------- A: classical control
print('\n=== A. classical control (same split, same subsample) ===')
t0 = time.time()
tv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000)
cv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60000)
ct = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                     max_features=40000)
Xa = sp.hstack([tv.fit_transform(SUB['T']), cv.fit_transform(SUB['C']),
                ct.fit_transform(SUB['T'])]).tocsr()
Xb = sp.hstack([tv.transform(VAL['T']), cv.transform(VAL['C']), ct.transform(VAL['T'])]).tocsr()
lr = LogisticRegression(max_iter=3000, class_weight='balanced', C=2.0, random_state=SEED)
lr.fit(Xa, SUB.label.values)
pA = lr.predict_proba(Xb)[:, 1]
fA, thA = best_macro_f1(pA, y_va)
apA = average_precision_score(1 - y_va, 1 - pA)
print(f'  classical: valF1={fA:.4f}@{thA:.2f} AP0={apA:.4f} '
      f'AUC0={roc_auc_score(1-y_va,1-pA):.4f}  [{time.time()-t0:.0f}s]')

print('\n=== B. Siamese CNN from scratch ===')
(bB, pB) = run(SiameseCNN(len(itos), A.dim), 'CNN')

print('\n=== C. Small Transformer from scratch ===')
(bC, pC) = run(SmallTransformer(len(itos), A.dim, L=max(A.tlen, A.clen)), 'TF')

print('\n' + '=' * 62)
print(f'{"model":<22}{"valF1":>9}{"AP class0":>12}')
print(f'{"A classical (control)":<22}{fA:>9.4f}{apA:>12.4f}')
print(f'{"B Siamese CNN":<22}{bB[0]:>9.4f}{bB[1]:>12.4f}')
print(f'{"C Small Transformer":<22}{bC[0]:>9.4f}{bC[1]:>12.4f}')
print(f'\nval class-0 base rate: {1-y_va.mean():.4f}')
print('DECISION RULE set before the run: neural passes only if it beats the')
print('classical control on this same split by >= +0.01 macro F1.')
print('=' * 62)

# ------------------------------------------------------- complementarity test
# The real question is not "does neural win" but "does neural add anything the
# classical model does not already have". Blend weights are tuned ON the
# validation set, so these numbers are an OPTIMISTIC UPPER BOUND, not a score.
print('\n--- complementarity (upper bound: blend tuned on val itself) ---')
print('rank correlation of class-0 scores:')
from scipy.stats import spearmanr
for nm, pp in [('CNN', pB), ('TF', pC)]:
    print(f'  classical vs {nm:<4} rho={spearmanr(1-pA, 1-pp).statistic:+.4f}')
print(f'  CNN       vs TF   rho={spearmanr(1-pB, 1-pC).statistic:+.4f}')
for nm, pp in [('CNN', pB), ('TF', pC)]:
    best = (fA, 0.0)
    for wgt in np.arange(0.0, 1.01, 0.05):
        f, _ = best_macro_f1((1 - wgt) * pA + wgt * pp, y_va)
        if f > best[0]: best = (f, wgt)
    print(f'  classical+{nm:<4} best blend F1={best[0]:.4f} @w={best[1]:.2f}  '
          f'(gain over classical {best[0]-fA:+.4f})')
print('=' * 62)
