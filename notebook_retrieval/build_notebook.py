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
# Penyisihan IFEST 2026 DAC — Retrieval + Cross-Encoder Pipeline

`Headline -> sentence retrieval (multilingual-e5-small) -> top-k body chunks ->
IndoBERT cross-encoder -> classification`, plus a TF-IDF sanity baseline, threshold
tuning, hard-negative analysis, optional feature fusion, and an optional
DeBERTa-v3-small benchmark.

## Adjustments made from the EDA (`ifest-2026-dac-comprehensive-eda-report`, full 14397/3603 run)

1. **No sentence-ending punctuation exists in `content` (0/14397 rows).** True sentence
   splitting is impossible. Retrieval below chunks article bodies into fixed ~35-word
   windows ("pseudo-sentences") instead — the EDA's sentence-evidence section already
   validated that evidence concentrates in a few such chunks (`max_sim` >> whole-document
   Jaccard), so this proxy is meaningful, not just a fallback.
2. **31% of validation content-hashes leak into train under a naive random split**
   (EDA Section 29). Grouping by `content_hash` (`GroupShuffleSplit`) is mandatory, not
   optional here.
3. **37% of test rows (1310/3603) share an exact `content_hash` with a train row**, and a
   further check below finds a meaningful number of **exact `(title, content)` duplicate
   pairs** across train/test. For those, the true label is already known from train —
   this is a value-add beyond the original spec: an exact-match lookup overrides the
   model's prediction only for that subset (Section 18).
4. **Titles are ~84.5% capitalized words** (Indonesian headline casing convention), so a
   naive capitalized-word NER proxy is not discriminative on titles. Entity features
   below are extracted from `content` (normal sentence case) and checked for coverage in
   the title instead — matching the EDA's Section 12 adaptation.
5. **Thousand separators are already stripped from the source numbers** (e.g. `"1 665"`
   not `"1,665"`), so token-level number matching is a noisy, approximate signal — kept
   as a weak feature, not a hard rule.
6. The EDA's tiny diagnostic classifiers topped out around **Macro F1 ~0.50** on
   lexical/relationship features alone (single split), which is the direct justification
   for escalating to a fine-tuned cross-encoder rather than stopping at TF-IDF.

## Runtime lessons carried over from this competition's completed CPU-baseline run

That notebook (`ifest-2026-dac-fast-tfidf-baseline`) hard-blew its 20-minute budget —
**59 minutes total** — because (a) char n-gram TF-IDF refit across 5 CV folds cost 22.5
min, and (b) CPU-only sentence-embedding encoding of ~28k texts cost 21 min. Two design
choices here directly respond to that:

- This notebook runs on **GPU (T4)**, and every embedding/fine-tuning step assumes it —
  CPU-only execution of the retrieval step in particular would be far worse here (~124k
  chunks to embed, not ~28k).
- No step here repeats an expensive fit across multiple CV folds — retrieval embeddings
  are computed once (Section 6) and the classifier uses a single `GroupShuffleSplit`
  (Section 4), not k-fold, exactly to avoid multiplying an expensive step 5x like that
  run did.

That run's B2 model also showed a LinearSVC fold collapse to Macro F1=0.317 (vs ~0.5
elsewhere) from mixing unscaled raw-magnitude features with normalized ones in one
linear model — Section 14's feature-fusion diagnostic applies `StandardScaler` to avoid
repeating that.
""")

# ============================================================
# 1. ENVIRONMENT
# ============================================================
md("## 1. Environment")

code("""
import os, sys, subprocess
def _pip(pkgs):
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs, check=True, timeout=180)
    except Exception as e:
        print(f"[warn] pip install {pkgs} failed/skipped: {e}")

_pip(['-U', 'transformers', 'accelerate'])
_pip(['sentencepiece', 'protobuf'])
""")

code("""
import re, time, json, hashlib, unicodedata, warnings, random
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)

import transformers
print("transformers:", transformers.__version__)

TIMINGS = {}
class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time()
        return self
    def __exit__(self, *a):
        dt = time.time() - self.t0
        TIMINGS[self.name] = TIMINGS.get(self.name, 0) + dt
        print(f"[TIMER] {self.name}: {dt:.1f}s")

RUN_T0 = time.time()

def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0
""")

# ============================================================
# 2. CONFIGURATION
# ============================================================
md("## 2. Configuration")

code("""
CONFIG = {
    # --- data ---
    'id_col': 'id', 'title_col': 'title', 'content_col': 'content', 'label_col': 'label',
    'seed': SEED,

    # --- retrieval ---
    'retriever_model': 'intfloat/multilingual-e5-small',
    'chunk_words': 35,          # pseudo-sentence window (no punctuation exists to split on -- see intro)
    'top_k': 6,                 # retrieved chunks per article (spec: 5-8)
    'retriever_max_len': 64,    # short chunks/titles, keeps encoding fast
    'retriever_batch_size': 256,

    # --- classifier ---
    'classifier_model': 'indobenchmark/indobert-base-p1',
    'alt_model': 'microsoft/deberta-v3-small',   # optional benchmark, Section 15
    'max_len': 320,             # title + up to top_k*chunk_words words, within 256-384 budget
    'batch_size': 16,
    'lr': 2e-5,
    'epochs': 3,
    'early_stopping_patience': 1,
    'val_size': 0.15,           # single GroupShuffleSplit (grouped CV would multiply GPU cost)

    # --- imbalance / thresholding ---
    'threshold_grid': [round(x, 2) for x in np.arange(0.20, 0.81, 0.05)],

    # --- run flags (kept conservative given this account's observed GPU-queue variance) ---
    'run_deberta_benchmark': True,
    'deberta_epochs': 2,
    'run_feature_fusion': True,
}
print(json.dumps({k: (v if not isinstance(v, np.ndarray) else list(v)) for k, v in CONFIG.items()}, indent=2, default=str))
""")

# ============================================================
# 3. LOAD AND INSPECT DATA
# ============================================================
md("## 3. Load and Inspect Data")

code("""
with Timer("data_loading"):
    DATA_FILES = {}
    for root, dirs, files in os.walk('/kaggle/input'):
        for f in files:
            if f in ('train.csv', 'test.csv', 'sample_submission.csv'):
                DATA_FILES.setdefault(f, os.path.join(root, f))
    print(DATA_FILES)
    assert len(DATA_FILES) == 3, f"Missing data files: {DATA_FILES}"
    train = pd.read_csv(DATA_FILES['train.csv'])
    test = pd.read_csv(DATA_FILES['test.csv'])
    sample_sub = pd.read_csv(DATA_FILES['sample_submission.csv'])

assert list(train.columns) == ['id','title','content','label']
assert list(test.columns) == ['id','title','content']

print("train shape:", train.shape, " test shape:", test.shape)
print("columns:", train.columns.tolist())
print("\\nmissing values (train):\\n", train.isna().sum())
print("\\nlabel distribution:\\n", train['label'].value_counts(normalize=True))

for df in (train, test):
    df['title_len'] = df['title'].str.len()
    df['content_len'] = df['content'].str.len()
print("\\ntitle/content length stats (train):")
print(train[['title_len','content_len']].describe())

print("\\nduplicate (title,content) rows (train):", train.duplicated(subset=['title','content']).sum())

def normalize_for_hash(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    s = re.sub(r'\\s+', ' ', s).strip()
    return s

train['content_hash'] = train['content'].apply(lambda s: hashlib.md5(normalize_for_hash(s).encode()).hexdigest())
train['title_hash'] = train['title'].apply(lambda s: hashlib.md5(normalize_for_hash(s).encode()).hexdigest())
test['content_hash'] = test['content'].apply(lambda s: hashlib.md5(normalize_for_hash(s).encode()).hexdigest())
test['title_hash'] = test['title'].apply(lambda s: hashlib.md5(normalize_for_hash(s).encode()).hexdigest())

print("duplicate article bodies (train):", train['content_hash'].duplicated().sum(),
      "/", train['content_hash'].nunique(), "unique bodies")
print("`id` is preserved for submission only -- never used as a model feature.")
""")

# ============================================================
# 4. LEAKAGE CHECK
# ============================================================
md("""
## 4. Leakage Check

No `article_id`/`body_id` column exists, so the group key is the **normalized-content
hash** computed above. A naive random split would let the same article body appear on
both sides of train/validation; the EDA measured this at **~31% contamination** under a
simulated random 80/20 split on this exact dataset. `GroupShuffleSplit` on
`content_hash` guarantees a body's rows all land on the same side, so the model can't
"cheat" by memorizing a body it already saw paired with a different headline during
training.
""")

code("""
from sklearn.model_selection import GroupShuffleSplit

with Timer("leakage_check"):
    # quick re-confirmation of the contamination this split method avoids
    rng = np.random.RandomState(SEED)
    sim = []
    for _ in range(10):
        shuf = train.sample(frac=1.0, random_state=rng.randint(0, 1_000_000))
        cut = int(len(shuf) * 0.85)
        tr_h, va_h = set(shuf['content_hash'].iloc[:cut]), set(shuf['content_hash'].iloc[cut:])
        sim.append(len(tr_h & va_h) / max(len(va_h), 1))
    print(f"simulated naive-random-split content contamination: {np.mean(sim)*100:.1f}%  -> using GroupShuffleSplit instead")

    gss = GroupShuffleSplit(n_splits=1, test_size=CONFIG['val_size'], random_state=SEED)
    train_idx, val_idx = next(gss.split(train, train['label'], groups=train['content_hash']))
    print(f"train rows: {len(train_idx)}   val rows: {len(val_idx)}")
    print(f"val label rate: {train['label'].iloc[val_idx].mean():.3f}  (train label rate: {train['label'].iloc[train_idx].mean():.3f})")

    overlap = set(train['content_hash'].iloc[train_idx]) & set(train['content_hash'].iloc[val_idx])
    assert len(overlap) == 0, "content_hash leaked across the split!"
    print("Confirmed: zero content_hash overlap between train_idx and val_idx.")
""")

# ============================================================
# 5. TEXT CLEANING
# ============================================================
md("## 5. Text Cleaning")

code("""
URL_RE = re.compile(r'https?://\\S+|www\\.\\S+')
HTML_RE = re.compile(r'<[^>]+>')
WS_RE = re.compile(r'\\s+')

def clean_text(s):
    s = unicodedata.normalize('NFKC', str(s))
    s = HTML_RE.sub(' ', s)
    s = URL_RE.sub(' ', s)
    s = WS_RE.sub(' ', s).strip()
    return s
    # Deliberately NOT done: stopword removal, stemming, number/punctuation stripping --
    # the EDA found none of these justified (negation words double as "stopwords",
    # numbers carry the factual-consistency signal, punctuation is already sparse).

with Timer("text_cleaning"):
    for df in (train, test):
        df['title_clean'] = df['title'].apply(clean_text)
        df['content_clean'] = df['content'].apply(clean_text)
print(train[['title','title_clean']].head(2))
""")

# ============================================================
# 6. SENTENCE RETRIEVAL
# ============================================================
md("""
## 6. Sentence Retrieval (multilingual-e5-small)

Article bodies are chunked into fixed-size pseudo-sentences (see intro), embedded with
`multilingual-e5-small`, and compared against the embedded headline via cosine
similarity to pick the top-k most relevant chunks per article. Embeddings are
**cached per unique `content_hash`/`title_hash`** across the combined train+test pool —
given 19% internal train duplication and 37% train/test content overlap, this avoids
re-encoding the same article many times.
""")

code("""
from transformers import AutoTokenizer, AutoModel

CHUNK_WORDS = CONFIG['chunk_words']

def chunk_pseudo_sentences(s, chunk_words=CHUNK_WORDS):
    words = s.split()
    if not words:
        return []
    return [' '.join(words[i:i+chunk_words]) for i in range(0, len(words), chunk_words)]

with Timer("retriever_load"):
    retr_tok = AutoTokenizer.from_pretrained(CONFIG['retriever_model'])
    retr_model = AutoModel.from_pretrained(CONFIG['retriever_model']).to(DEVICE).eval()

def average_pool(last_hidden, attention_mask):
    last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

@torch.no_grad()
def encode_texts(texts, prefix, batch_size=None, max_length=None):
    batch_size = batch_size or CONFIG['retriever_batch_size']
    max_length = max_length or CONFIG['retriever_max_len']
    out_embs = []
    use_amp = torch.cuda.is_available()
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i:i+batch_size]]
        enc = retr_tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(DEVICE)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            out = retr_model(**enc)
        emb = average_pool(out.last_hidden_state, enc['attention_mask'])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        out_embs.append(emb.float().cpu().numpy())
    return np.concatenate(out_embs, axis=0) if out_embs else np.zeros((0, retr_model.config.hidden_size))
""")

code("""
with Timer("retrieval_encode_and_select"):
    combined = pd.concat([
        train[['content_hash','title_hash','content_clean','title_clean']],
        test[['content_hash','title_hash','content_clean','title_clean']],
    ], ignore_index=True)

    # --- unique content bodies -> chunk + encode once ---
    uniq_content = combined.drop_duplicates('content_hash')[['content_hash','content_clean']].reset_index(drop=True)
    chunk_lists = uniq_content['content_clean'].apply(chunk_pseudo_sentences)
    flat_chunks, chunk_ranges = [], {}
    for h, chunks in zip(uniq_content['content_hash'], chunk_lists):
        start = len(flat_chunks)
        flat_chunks.extend(chunks if chunks else [''])
        chunk_ranges[h] = (start, len(flat_chunks))
    print(f"unique articles: {len(uniq_content)}  total pseudo-sentence chunks: {len(flat_chunks)}")

    chunk_embs_flat = encode_texts(flat_chunks, prefix='passage: ')

    # --- unique titles -> encode once ---
    uniq_title = combined.drop_duplicates('title_hash')[['title_hash','title_clean']].reset_index(drop=True)
    title_embs = encode_texts(uniq_title['title_clean'].tolist(), prefix='query: ', max_length=32)
    title_emb_map = dict(zip(uniq_title['title_hash'], title_embs))

print(f"chunk embedding matrix: {chunk_embs_flat.shape}  title embedding matrix: {title_embs.shape}")
""")

code("""
def retrieve_top_k(df, top_k=None):
    top_k = top_k or CONFIG['top_k']
    retrieved_texts, max_sims, mean_topk_sims = [], [], []
    for h, th in zip(df['content_hash'], df['title_hash']):
        start, end = chunk_ranges[h]
        c_embs = chunk_embs_flat[start:end]
        c_texts = flat_chunks[start:end]
        t_emb = title_emb_map[th]
        sims = c_embs @ t_emb
        k = min(top_k, len(sims))
        top_idx = np.argsort(-sims)[:k]
        retrieved_texts.append(' '.join(c_texts[i] for i in sorted(top_idx)))
        max_sims.append(float(sims.max()) if len(sims) else 0.0)
        mean_topk_sims.append(float(sims[top_idx].mean()) if len(top_idx) else 0.0)
    return retrieved_texts, np.array(max_sims), np.array(mean_topk_sims)

with Timer("retrieval_apply"):
    train['retrieved_text'], train['max_chunk_sim'], train['mean_topk_sim'] = retrieve_top_k(train)
    test['retrieved_text'], test['max_chunk_sim'], test['mean_topk_sim'] = retrieve_top_k(test)

print(train[['title','retrieved_text','max_chunk_sim']].head(2).to_string())
del chunk_embs_flat  # free ~hundreds of MB once retrieval is baked into the dataframes
""")

# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================
md("""
## 7. Feature Engineering (lightweight consistency features)

All features are a deterministic function of `(title, content)` text — computed once
globally (no label leakage) and reused by the baseline, the fusion experiment, and the
hard-negative analysis.
""")

code("""
NUM_RE = re.compile(r'\\d+[.,]?\\d*\\s*%?')
YEAR_RE = re.compile(r'\\b(?:19|20)\\d{2}\\b')
NEGATION_WORDS = ['tidak','bukan','belum','tanpa','gagal','ditolak','membantah','bantah','sangkal','menyangkal','menolak']

def word_list(s):
    return re.findall(r\"[\\w']+\", str(s).lower())

def safe_div(a, b):
    return a / b if b else 0.0

def extract_caps_entities(s):
    import string as _s
    sentences = re.split(r'(?<=[.!?])\\s+', str(s)) or [str(s)]
    ents = []
    for sent in sentences:
        toks = sent.split()
        for i, t in enumerate(toks):
            core = t.strip(_s.punctuation)
            if i > 0 and core[:1].isupper() and core.isalpha() and len(core) > 2:
                ents.append(core.lower())
    return set(ents)

def build_relationship_features(df):
    feats = []
    for title, content in zip(df['title_clean'], df['content_clean']):
        tw_l, cw_l = word_list(title), word_list(content)
        tw, cw = set(tw_l), set(cw_l)
        inter, union = tw & cw, tw | cw
        tnum, cnum = set(NUM_RE.findall(title)), set(NUM_RE.findall(content))
        tyear, cyear = set(YEAR_RE.findall(title)), set(YEAR_RE.findall(content))
        c_ents = extract_caps_entities(content)
        title_l = title.lower()
        ent_in_title = sum(1 for e in c_ents if e in title_l)
        f = {
            'n_shared_words': len(inter),
            'title_coverage': safe_div(len(inter), len(tw)),
            'jaccard': safe_div(len(inter), len(union)),
            'title_len': len(title), 'content_len': len(content),
            'title_word_count': len(tw_l), 'content_word_count': len(cw_l),
            'char_ratio': safe_div(len(title), len(content)),
            'n_numbers_title': len(tnum), 'n_numbers_content': len(cnum),
            'number_overlap': safe_div(len(tnum & cnum), len(tnum)) if tnum else 1.0,
            'number_mismatch': float(len(tnum) > 0 and len(tnum & cnum) == 0),
            'n_years_title': len(tyear),
            'date_overlap': safe_div(len(tyear & cyear), len(tyear)) if tyear else 1.0,
            'content_entity_count': len(c_ents),
            'entity_coverage_in_title': safe_div(ent_in_title, len(c_ents)) if c_ents else np.nan,
            'title_negation': float(any(w in tw_l for w in NEGATION_WORDS)),
            'content_negation': float(any(w in cw_l for w in NEGATION_WORDS)),
        }
        f['negation_mismatch'] = float(f['title_negation'] != f['content_negation'])
        feats.append(f)
    return pd.DataFrame(feats, index=df.index).fillna(0.0)

with Timer("feature_engineering"):
    train_feats = build_relationship_features(train)
    test_feats = build_relationship_features(test)
FEATURE_COLS = list(train_feats.columns)
print(f"{len(FEATURE_COLS)} relationship features built:", FEATURE_COLS)
print(pd.concat([train_feats, train['label']], axis=1).groupby('label').mean().T)
""")

# ============================================================
# 8. BASELINE: TF-IDF + LOGISTIC REGRESSION
# ============================================================
md("## 8. Baseline — TF-IDF + Logistic Regression (sanity check)")

code("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import scipy.sparse as sp

with Timer("baseline_tfidf"):
    title_vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=20000)
    content_vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=40000)

    Xtr_title = title_vec.fit_transform(train['title_clean'].iloc[train_idx])
    Xva_title = title_vec.transform(train['title_clean'].iloc[val_idx])
    Xtr_content = content_vec.fit_transform(train['content_clean'].iloc[train_idx])
    Xva_content = content_vec.transform(train['content_clean'].iloc[val_idx])

    Xtr = sp.hstack([Xtr_title, Xtr_content]).tocsr()
    Xva = sp.hstack([Xva_title, Xva_content]).tocsr()
    ytr, yva = train['label'].iloc[train_idx].values, train['label'].iloc[val_idx].values

    baseline_clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED)
    baseline_clf.fit(Xtr, ytr)
    base_pred = baseline_clf.predict(Xva)

baseline_macro_f1 = f1_score(yva, base_pred, average='macro')
print(f"Baseline TF-IDF+LogReg Macro F1: {baseline_macro_f1:.4f}")
print(f"F1 class0: {f1_score(yva, base_pred, pos_label=0):.4f}  F1 class1: {f1_score(yva, base_pred, pos_label=1):.4f}")
print("Confusion matrix:\\n", confusion_matrix(yva, base_pred))
print(classification_report(yva, base_pred, digits=4))
""")

# ============================================================
# 9. TRANSFORMER MODEL (IndoBERT cross-encoder)
# ============================================================
md("""
## 9 & 10. Transformer Model — IndoBERT Cross-Encoder (class-weighted, FP16, early stopping)

Input: `[TITLE] [SEP] [TOP-K RETRIEVED BODY CHUNKS]`. Class imbalance (~9:1, confirmed by
EDA) is handled with a **class-weighted cross-entropy loss** (inverse label frequency)
rather than resampling, since it composes cleanly with `Trainer` and avoids duplicating
the (already large) tokenized minority-class inputs in memory.
""")

code("""
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, Trainer, TrainingArguments,
                           EarlyStoppingCallback)

class PairDataset(Dataset):
    def __init__(self, titles, bodies, labels, tokenizer, max_len):
        self.titles = list(titles); self.bodies = list(bodies)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self):
        return len(self.titles)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.titles[idx], self.bodies[idx], truncation=True,
                              max_length=self.max_len, padding='max_length', return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'macro_f1': f1_score(labels, preds, average='macro'),
        'f1_class0': f1_score(labels, preds, pos_label=0, zero_division=0),
        'f1_class1': f1_score(labels, preds, pos_label=1, zero_division=0),
        'accuracy': float((preds == labels).mean()),
    }

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

def make_class_weights(labels):
    n = len(labels); n0 = (labels==0).sum(); n1 = (labels==1).sum()
    w0, w1 = n/(2*n0), n/(2*n1)
    return torch.tensor([w0, w1], dtype=torch.float)

def train_cross_encoder(model_name, train_titles, train_bodies, train_labels,
                         val_titles, val_bodies, val_labels, epochs, out_dir, max_len=None,
                         early_stopping=True):
    # early_stopping=False is used for the final full-data retrain (Section 16): there is
    # no legitimate held-out set left at that point, and transformers' EarlyStoppingCallback
    # asserts load_best_model_at_end=True at train start -- so early stopping and eval are
    # fully disabled here rather than toggled off after the Trainer is already built.
    max_len = max_len or CONFIG['max_len']
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_ds = PairDataset(train_titles, train_bodies, train_labels, tok, max_len)
    class_weights = make_class_weights(np.array(train_labels))

    if early_stopping:
        val_ds = PairDataset(val_titles, val_bodies, val_labels, tok, max_len)
        args = TrainingArguments(
            output_dir=out_dir, num_train_epochs=epochs,
            per_device_train_batch_size=CONFIG['batch_size'], per_device_eval_batch_size=CONFIG['batch_size']*2,
            learning_rate=CONFIG['lr'], weight_decay=0.01,
            eval_strategy='epoch', save_strategy='epoch', save_total_limit=1,
            load_best_model_at_end=True, metric_for_best_model='macro_f1', greater_is_better=True,
            fp16=torch.cuda.is_available(), dataloader_num_workers=0, seed=SEED,
            logging_steps=100, report_to=[], disable_tqdm=False,
        )
        trainer = WeightedTrainer(
            model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
            compute_metrics=compute_metrics, class_weights=class_weights,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=CONFIG['early_stopping_patience'])],
        )
    else:
        args = TrainingArguments(
            output_dir=out_dir, num_train_epochs=epochs,
            per_device_train_batch_size=CONFIG['batch_size'],
            learning_rate=CONFIG['lr'], weight_decay=0.01,
            eval_strategy='no', save_strategy='no',
            fp16=torch.cuda.is_available(), dataloader_num_workers=0, seed=SEED,
            logging_steps=100, report_to=[], disable_tqdm=False,
        )
        trainer = WeightedTrainer(
            model=model, args=args, train_dataset=train_ds,
            class_weights=class_weights,
        )
    return trainer, tok
""")

code("""
with Timer("indobert_training"):
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    indobert_trainer, indobert_tok = train_cross_encoder(
        CONFIG['classifier_model'],
        train['title_clean'].iloc[train_idx], train['retrieved_text'].iloc[train_idx], train['label'].iloc[train_idx].values,
        train['title_clean'].iloc[val_idx], train['retrieved_text'].iloc[val_idx], train['label'].iloc[val_idx].values,
        epochs=CONFIG['epochs'], out_dir='/kaggle/working/indobert_ckpt',
    )
    indobert_trainer.train()
    indobert_train_time = time.time() - t0
    indobert_gpu_mem = gpu_mem_mb()
print(f"IndoBERT training time: {indobert_train_time:.1f}s   peak GPU mem: {indobert_gpu_mem:.0f} MB")
""")

# ============================================================
# 11. VALIDATION
# ============================================================
md("## 11. Validation")

code("""
with Timer("indobert_validation"):
    val_out = indobert_trainer.predict(indobert_trainer.eval_dataset)
    val_logits = val_out.predictions
    val_labels = train['label'].iloc[val_idx].values
    val_probs = torch.softmax(torch.tensor(val_logits), dim=-1).numpy()[:, 1]
    val_pred_default = (val_probs >= 0.5).astype(int)

print(f"Macro F1 (threshold=0.5): {f1_score(val_labels, val_pred_default, average='macro'):.4f}")
print(f"F1 class0: {f1_score(val_labels, val_pred_default, pos_label=0):.4f}")
print(f"F1 class1: {f1_score(val_labels, val_pred_default, pos_label=1):.4f}")
print(f"Accuracy: {(val_pred_default==val_labels).mean():.4f}")
print("Confusion matrix:\\n", confusion_matrix(val_labels, val_pred_default))
""")

# ============================================================
# 12. THRESHOLD TUNING
# ============================================================
md("""
## 12. Threshold Tuning

The EDA's own quick diagnostic classifier already showed threshold=0.5 is suboptimal for
Macro F1 under this imbalance (best around 0.4 for that proxy model) — we do not assume
0.5 here either, and instead sweep and pick empirically on the validation set.
""")

code("""
with Timer("threshold_tuning"):
    thr_rows = []
    for thr in CONFIG['threshold_grid']:
        pred = (val_probs >= thr).astype(int)
        thr_rows.append({
            'threshold': thr,
            'f1_class0': f1_score(val_labels, pred, pos_label=0, zero_division=0),
            'f1_class1': f1_score(val_labels, pred, pos_label=1, zero_division=0),
            'macro_f1': f1_score(val_labels, pred, average='macro'),
            'pos_rate': pred.mean(),
        })
    thr_df = pd.DataFrame(thr_rows)
    print(thr_df.to_string(index=False))
    BEST_THRESHOLD = float(thr_df.loc[thr_df['macro_f1'].idxmax(), 'threshold'])
    BEST_VAL_MACRO_F1 = float(thr_df['macro_f1'].max())
print(f"\\nBest threshold: {BEST_THRESHOLD}  (Macro F1={BEST_VAL_MACRO_F1:.4f} vs {f1_score(val_labels, val_pred_default, average='macro'):.4f} at 0.5)")
""")

# ============================================================
# 13. HARD NEGATIVE ANALYSIS
# ============================================================
md("## 13. Hard Negative Analysis")

code("""
with Timer("hard_negative_analysis"):
    val_pred_best = (val_probs >= BEST_THRESHOLD).astype(int)
    # drop the quick-inspection length columns from Section 3 first -- they share names
    # with build_relationship_features' output ('title_len','content_len') and would
    # otherwise make .join() raise "columns overlap but no suffix specified"
    val_df = train.iloc[val_idx].drop(columns=['title_len', 'content_len'], errors='ignore').copy()
    val_df['pred'] = val_pred_best
    val_df['prob'] = val_probs
    val_df = val_df.join(train_feats.iloc[val_idx])

    mistakes = val_df[val_df['label'] != val_df['pred']].copy()
    high_sim_wrong = mistakes[(mistakes['label']==0) & (mistakes['max_chunk_sim'] > mistakes['max_chunk_sim'].median())]
    print(f"total validation mistakes: {len(mistakes)} / {len(val_df)}")
    print(f"high-similarity false positives (true=0, high retrieval similarity, model likely fooled by overlap): {len(high_sim_wrong)}")

    pd.set_option('display.max_colwidth', 140)
    cols = ['title','retrieved_text','label','pred','prob','max_chunk_sim','number_mismatch','date_overlap','entity_coverage_in_title','negation_mismatch']
    print("\\n=== Representative hard negatives (high similarity, wrongly predicted) ===")
    display_cols = [c for c in cols if c in high_sim_wrong.columns]
    print(high_sim_wrong[display_cols].head(15).to_string())
""")

# ============================================================
# 14. OPTIONAL FEATURE FUSION
# ============================================================
md("""
## 14. Optional Feature Fusion

Quick diagnostic (not a rigorous nested CV): split the validation set itself into a small
fusion-train/fusion-test to check whether concatenating the handcrafted relationship
features with IndoBERT's predicted probability improves Macro F1 over IndoBERT alone.
""")

code("""
FUSION_IMPROVEMENT = 0.0
if CONFIG['run_feature_fusion']:
    with Timer("feature_fusion"):
        from sklearn.model_selection import train_test_split as _tts
        from sklearn.preprocessing import StandardScaler
        # NOTE: an earlier CPU-baseline run on this same competition showed a LinearSVC
        # fold collapse to Macro F1=0.317 (vs ~0.47-0.53 elsewhere) when raw-scale pair
        # features (e.g. content_len in the thousands) were hstacked next to normalized
        # TF-IDF/probability features without scaling. StandardScaler here avoids
        # repeating that instability in this linear fusion model.
        fz_idx = np.arange(len(val_df))
        fz_tr, fz_te = _tts(fz_idx, test_size=0.4, random_state=SEED, stratify=val_df['label'])

        X_indobert_only = val_df[['prob']].values
        X_fused = pd.concat([val_df[['prob']], val_df[FEATURE_COLS]], axis=1).values
        y_fz = val_df['label'].values

        f1_only, f1_fused = [], []
        for X, store in [(X_indobert_only, f1_only), (X_fused, f1_fused)]:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X[fz_tr])
            X_te_s = scaler.transform(X[fz_te])
            clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED)
            clf.fit(X_tr_s, y_fz[fz_tr])
            pred = clf.predict(X_te_s)
            store.append(f1_score(y_fz[fz_te], pred, average='macro'))

        print(f"IndoBERT-prob-only Macro F1 (diagnostic split): {f1_only[0]:.4f}")
        print(f"IndoBERT-prob + relationship features Macro F1 (diagnostic split): {f1_fused[0]:.4f}")
        FUSION_IMPROVEMENT = f1_fused[0] - f1_only[0]
        if FUSION_IMPROVEMENT >= 0.01:
            print(f"-> Fusion improves Macro F1 by {FUSION_IMPROVEMENT:+.4f} (>=0.01): worth keeping.")
        else:
            print(f"-> Improvement ({FUSION_IMPROVEMENT:+.4f}) is negligible: keep the simpler IndoBERT-only model.")
else:
    print("Feature fusion diagnostic skipped (CONFIG['run_feature_fusion']=False).")
""")

# ============================================================
# 15. OPTIONAL DEBERTA BENCHMARK
# ============================================================
md("""
## 15. Optional DeBERTa-v3-small Benchmark

Same retrieval-augmented input, same training recipe, fewer epochs by default
(`CONFIG['deberta_epochs']`) purely to bound GPU time for a comparison that is explicitly
optional. Wrapped in `try/except` so a tokenizer/dependency hiccup can't take down the
rest of the notebook.
""")

code("""
deberta_results = None
if CONFIG['run_deberta_benchmark']:
    try:
        with Timer("deberta_benchmark"):
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            deberta_trainer, deberta_tok = train_cross_encoder(
                CONFIG['alt_model'],
                train['title_clean'].iloc[train_idx], train['retrieved_text'].iloc[train_idx], train['label'].iloc[train_idx].values,
                train['title_clean'].iloc[val_idx], train['retrieved_text'].iloc[val_idx], train['label'].iloc[val_idx].values,
                epochs=CONFIG['deberta_epochs'], out_dir='/kaggle/working/deberta_ckpt',
            )
            deberta_trainer.train()
            deberta_train_time = time.time() - t0
            deberta_gpu_mem = gpu_mem_mb()

            t0 = time.time()
            d_out = deberta_trainer.predict(deberta_trainer.eval_dataset)
            deberta_infer_time = time.time() - t0
            d_probs = torch.softmax(torch.tensor(d_out.predictions), dim=-1).numpy()[:, 1]
            d_pred = (d_probs >= 0.5).astype(int)
            deberta_macro_f1 = f1_score(val_labels, d_pred, average='macro')

        deberta_results = {'model': CONFIG['alt_model'], 'val_macro_f1': deberta_macro_f1,
                            'train_time_s': deberta_train_time, 'infer_time_s': deberta_infer_time,
                            'gpu_mem_mb': deberta_gpu_mem}
        print(deberta_results)
    except Exception as e:
        print(f"[warn] DeBERTa benchmark failed/skipped: {e}")
else:
    print("DeBERTa benchmark skipped (CONFIG['run_deberta_benchmark']=False).")
""")

code("""
with Timer("indobert_infer_timing"):
    t0 = time.time()
    _ = indobert_trainer.predict(indobert_trainer.eval_dataset)
    indobert_infer_time = time.time() - t0

compare_rows = [{'model': CONFIG['classifier_model'], 'val_macro_f1': BEST_VAL_MACRO_F1,
                  'train_time_s': indobert_train_time, 'infer_time_s': indobert_infer_time,
                  'gpu_mem_mb': indobert_gpu_mem}]
if deberta_results is not None:
    compare_rows.append(deberta_results)
compare_df = pd.DataFrame(compare_rows)
print(compare_df.to_string(index=False))

if deberta_results is not None and deberta_results['val_macro_f1'] > BEST_VAL_MACRO_F1 + 0.01 and deberta_results['train_time_s'] < indobert_train_time * 2:
    BEST_ARCHITECTURE = CONFIG['alt_model']
    print(f"\\n-> Selecting {BEST_ARCHITECTURE}: meaningfully better Macro F1 without a disproportionate runtime cost.")
else:
    BEST_ARCHITECTURE = CONFIG['classifier_model']
    print(f"\\n-> Keeping {BEST_ARCHITECTURE}: best accuracy/compute trade-off (see Section 15 rule: don't switch for a marginal gain).")
""")

# ============================================================
# 16. FINAL TRAINING
# ============================================================
md("""
## 16. Final Training (full data, selected architecture)

Retrain on **all** of `train` (train_idx + val_idx combined) with the architecture chosen
above and the same hyperparameters. There is no held-out set left at this point, so we
train for a fixed number of epochs (the number of epochs the validation run actually
completed before early stopping) rather than re-running early stopping against nothing.
""")

code("""
with Timer("final_training"):
    t0 = time.time()
    final_epochs = max(1, int(round(indobert_trainer.state.epoch or CONFIG['epochs'])))
    print(f"Final training epochs (matched to validated run): {final_epochs}")

    all_labels = train['label'].values
    final_trainer, final_tok = train_cross_encoder(
        BEST_ARCHITECTURE,
        train['title_clean'], train['retrieved_text'], all_labels,
        None, None, None,
        epochs=final_epochs, out_dir='/kaggle/working/final_ckpt',
        early_stopping=False,
    )
    final_trainer.train()
    final_train_time = time.time() - t0
print(f"Final model trained on {len(train)} rows in {final_train_time:.1f}s")
""")

# ============================================================
# 17. TEST PREDICTION
# ============================================================
md("## 17. Test Prediction")

code("""
with Timer("test_inference"):
    t0 = time.time()
    test_ds = PairDataset(test['title_clean'], test['retrieved_text'], None, final_tok, CONFIG['max_len'])
    test_out = final_trainer.predict(test_ds)
    test_probs = torch.softmax(torch.tensor(test_out.predictions), dim=-1).numpy()[:, 1]
    test_pred = (test_probs >= BEST_THRESHOLD).astype(int)
    final_infer_time = time.time() - t0
print(f"Test inference done in {final_infer_time:.1f}s. Positive-rate: {test_pred.mean():.3f}")
""")

# ============================================================
# 18. SUBMISSION
# ============================================================
md("""
## 18. Submission (with EDA-driven exact-match override)

Beyond the base spec: for test rows whose exact `(title_hash, content_hash)` pair also
appears in train, the true label is already known — override the model's prediction with
it. This directly uses the EDA's finding that a large share of test content is reused
verbatim from train, and is a legitimate use of train labels (not test-label leakage).
""")

code("""
with Timer("submission"):
    exact_pair_map = (train.drop_duplicates(subset=['title_hash','content_hash'])
                            .set_index(['title_hash','content_hash'])['label'].to_dict())
    override_mask = list(zip(test['title_hash'], test['content_hash']))
    n_overridden = sum(1 for k in override_mask if k in exact_pair_map)
    print(f"exact (title,content) pair matches found in train for {n_overridden}/{len(test)} test rows "
          f"({n_overridden/len(test)*100:.1f}%) -- overriding model prediction for these with the known train label.")

    final_pred = test_pred.copy()
    for i, key in enumerate(override_mask):
        if key in exact_pair_map:
            final_pred[i] = exact_pair_map[key]

    submission = pd.DataFrame({'id': test['id'], 'label': final_pred.astype(int)})
    submission = submission.set_index('id').loc[test['id']].reset_index()

    assert submission.shape[0] == len(test)
    assert set(submission['id']) == set(sample_sub['id'])
    assert set(submission['label'].unique()).issubset({0,1})
    assert submission['label'].isna().sum() == 0
    assert list(submission.columns) == ['id','label']

    submission.to_csv('/kaggle/working/submission.csv', index=False)

print(submission.head())
print(submission.shape)
print(submission['label'].value_counts(normalize=True))
""")

# ============================================================
# FINAL OUTPUT
# ============================================================
md("## Final Runtime and Results Summary")

code("""
TOTAL_RUNTIME = time.time() - RUN_T0
print("=" * 50)
print("Best Model:", BEST_ARCHITECTURE)
print(f"Best Validation Macro F1: {BEST_VAL_MACRO_F1:.4f}")
print(f"Best Threshold: {BEST_THRESHOLD}")
print(f"Training Time: {indobert_train_time + (deberta_results['train_time_s'] if deberta_results else 0) + final_train_time:.1f}s "
      f"(validation run + optional benchmark + final full-data retrain)")
print(f"Approximate GPU Memory: {max(indobert_gpu_mem, (deberta_results or {}).get('gpu_mem_mb', 0)):.0f} MB")
print(f"Submission Shape: {submission.shape}")
print("Submission Path: /kaggle/working/submission.csv")
print("=" * 50)

print(f"\\nBaseline (TF-IDF+LogReg) Macro F1 was {baseline_macro_f1:.4f} -- transformer improvement: "
      f"{BEST_VAL_MACRO_F1 - baseline_macro_f1:+.4f}")
print(f"Feature-fusion improvement (diagnostic): {FUSION_IMPROVEMENT:+.4f}")
print(f"Exact-match override applied to {n_overridden} test rows.")
print(f"\\nTotal notebook runtime: {TOTAL_RUNTIME:.1f}s ({TOTAL_RUNTIME/60:.1f} min)")
print("\\nPer-section timings:")
for k, v in sorted(TIMINGS.items(), key=lambda kv: -kv[1]):
    print(f"  {k}: {v:.1f}s")
""")

nb['cells'] = cells
nb['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb['metadata']['language_info'] = {'name': 'python', 'version': '3.11'}

with open('ifest2026_dac_retrieval_crossencoder.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook written.")
