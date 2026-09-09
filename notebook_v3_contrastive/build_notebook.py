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
# Penyisihan IFEST 2026 DAC — v3: Hybrid Retrieval + Same-Content Contrastive Fine-Tuning

v2 of this pipeline reached **Macro F1 = 0.7484** (cold-start validation) / **~0.73 leaderboard**.
Post-mortem on that run's confusion matrix (`F1 class0=0.543` vs `F1 class1=0.954`) and its
hard-negative examples showed the bottleneck precisely: **the model learned "does this
look topically similar?" instead of "does this specific claim hold?"** — e.g. a title
claiming crime dropped in *Jatim* scored 0.90+ similarity against a body about *Jateng*,
because the retriever optimizes for semantic proximity, not claim verification. The
threshold sweep (flat ~0.72-0.75 across 0.2-0.7) also confirms this is a **representation
problem, not a threshold/imbalance problem**.

## What changed in v3 (mapped to the priority list)

| # | Problem | Fix in this notebook |
|---|---|---|
| 1 | Hard-negative/contradiction learning | **Phase 2: same-content contrastive fine-tuning** (Section 11) — for every `content_hash` with both label=1 and label=0 titles, force `score(title_pos, content) > score(title_neg, content)` via margin ranking loss |
| 2 | Retrieval information loss | **Overlapping 50-word/stride-25 chunks + hybrid retrieval** (top-3 E5 semantic + top-2 TF-IDF lexical + lede chunk), replacing non-overlapping top-6-semantic-only (Section 6) |
| 3 | Same-content pairs unexploited | Directly mined into contrastive pairs (Section 10) instead of being treated as independent rows |
| 4 | Validation doesn't match test distribution | **EXP1**: report cold-start Macro F1 *and* a simulated competition-realistic estimate that accounts for test's exact-pair overlap rate (Section 13) |
| 5 | Entity/number/action mismatch underused | Explicit **conflict-text** string (entities/numbers in body but not title) appended to the model's input text (Section 8) |
| 6 | Overfitting (train loss down, val loss up) | Unchanged early stopping + class weighting, now paired with the contrastive phase which targets the actual failure mode rather than more epochs of the same objective |
| — | DeBERTa benchmark crashed (`expected scalar type Half but found Float`) | fp16 disabled specifically for DeBERTa-v3 (known disentangled-attention/AMP incompatibility); IndoBERT keeps fp16 |

## What was deliberately scoped out (told plainly, not silently dropped)

- **Cross-article synthetic hard-negative mining** (mining a hard negative title from a
  *different* article with high E5 similarity) — only same-content pairs are used here.
  Worth adding next if Phase 2 shows the contrastive idea works at all.
- **Full structured evidence/counter-evidence composition** (separate "supporting" vs
  "contradicting" chunk channels) — simplified to hybrid retrieval + one conflict-text
  string, since a full dual-channel architecture is a much larger change to validate
  safely in one pass.

## Safety net

Phase 2 (contrastive) is a genuinely new idea on ~1-2k mined pairs — it could just as
easily overfit or catastrophically forget as help. This notebook explicitly **compares
Phase 1-only vs Phase 1+2 validation Macro F1 and keeps whichever is better**, so a
failed hypothesis about contrastive learning degrades gracefully to v2's known-good
result rather than silently shipping a worse model.
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
import re, time, json, hashlib, unicodedata, warnings, random, string as _string
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

    # --- retrieval (v3: overlapping + hybrid) ---
    'retriever_model': 'intfloat/multilingual-e5-small',
    'chunk_words': 50,          # window size
    'chunk_stride': 25,         # 50% overlap -- avoids splitting "tidak akan melakukan" across chunks
    'top_k_semantic': 3,        # E5 cosine-similarity chunks
    'top_k_lexical': 2,         # TF-IDF lexical-overlap chunks (catches exact entity/number strings E5 may de-prioritize)
    'include_first_chunk': True,  # lede -- inverted-pyramid Indonesian news usually fronts the key fact
    'retriever_max_len': 96,    # chunks are now longer (50 words) than v2's 35
    'retriever_batch_size': 256,
    'lexical_max_features': 60000,

    # --- classifier ---
    'classifier_model': 'indobenchmark/indobert-base-p1',
    'alt_model': 'microsoft/deberta-v3-small',
    'max_len': 384,             # bumped from v2's 320: hybrid selection can pick up to 6 chunks + conflict text
    'batch_size': 16,
    'lr': 2e-5,
    'epochs': 3,
    'early_stopping_patience': 1,
    'val_size': 0.15,

    # --- Phase 2: same-content contrastive fine-tuning ---
    'contrastive_epochs': 2,
    'contrastive_lr': 1e-5,
    'contrastive_margin': 0.15,
    'contrastive_batch_size': 8,

    # --- imbalance / thresholding ---
    'threshold_grid': [round(x, 2) for x in np.arange(0.20, 0.81, 0.05)],

    # --- run flags ---
    'run_deberta_benchmark': True,
    'deberta_epochs': 2,
    'run_feature_fusion': True,
}
print(json.dumps({k: v for k, v in CONFIG.items()}, indent=2, default=str))
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
print("label distribution:\\n", train['label'].value_counts(normalize=True))

def normalize_for_hash(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    s = re.sub(r'\\s+', ' ', s).strip()
    return s

for df in (train, test):
    df['content_hash'] = df['content'].apply(lambda s: hashlib.md5(normalize_for_hash(s).encode()).hexdigest())
    df['title_hash'] = df['title'].apply(lambda s: hashlib.md5(normalize_for_hash(s).encode()).hexdigest())

print("duplicate article bodies (train):", train['content_hash'].duplicated().sum(),
      "/", train['content_hash'].nunique(), "unique bodies")
print("`id` is preserved for submission only -- never used as a model feature.")
""")

md("### 3.1 Exact train/test pair overlap (needed early for the EXP1 validation diagnostic below)")

code("""
with Timer("exact_pair_overlap_check"):
    exact_pair_map = (train.drop_duplicates(subset=['title_hash','content_hash'])
                            .set_index(['title_hash','content_hash'])['label'].to_dict())
    test_override_keys = list(zip(test['title_hash'], test['content_hash']))
    n_overridden_test = sum(1 for k in test_override_keys if k in exact_pair_map)
    TEST_OVERRIDE_RATE = n_overridden_test / len(test)
    print(f"test rows with an exact (title,content) match in train: {n_overridden_test}/{len(test)} "
          f"({TEST_OVERRIDE_RATE*100:.1f}%) -- these get their true label from train directly at submission time (Section 18).")
""")

# ============================================================
# 4. LEAKAGE CHECK
# ============================================================
md("""
## 4. Leakage Check

Grouping by `content_hash` remains the correct methodology for *training* (a body must
not appear on both sides of the split) — the point raised about validation not matching
test distribution is addressed separately as a diagnostic in Section 13, not by weakening
this split.
""")

code("""
from sklearn.model_selection import GroupShuffleSplit

with Timer("leakage_check"):
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

with Timer("text_cleaning"):
    for df in (train, test):
        df['title_clean'] = df['title'].apply(clean_text)
        df['content_clean'] = df['content'].apply(clean_text)
print(train[['title','title_clean']].head(2))
""")

# ============================================================
# SHARED TEXT UTILITIES (moved up so retrieval-coverage diagnostics can use them)
# ============================================================
md("""
## 5.1 Shared Text Utilities

Number/entity extraction and word-overlap helpers, used by the retrieval-coverage
diagnostic (Section 7), feature engineering (Section 8), and the conflict-text builder.
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
    sentences = re.split(r'(?<=[.!?])\\s+', str(s)) or [str(s)]
    ents = []
    for sent in sentences:
        toks = sent.split()
        for i, t in enumerate(toks):
            core = t.strip(_string.punctuation)
            if i > 0 and core[:1].isupper() and core.isalpha() and len(core) > 2:
                ents.append(core.lower())
    return set(ents)
""")

# ============================================================
# 6. HYBRID OVERLAPPING RETRIEVAL
# ============================================================
md("""
## 6. Hybrid Overlapping Retrieval (multilingual-e5-small + TF-IDF)

Two changes from v2:
1. **Overlapping chunks** (50 words, stride 25) instead of non-overlapping 35-word blocks
   — a fact split across a chunk boundary in v2 (e.g. subject in chunk *n*, negation in
   chunk *n+1*) now has a good chance of appearing whole in at least one window.
2. **Hybrid selection**: top-3 chunks by E5 semantic similarity + top-2 by TF-IDF lexical
   overlap (catches exact entity/number strings that a paraphrase-oriented embedding can
   under-rank) + the first chunk (lede) always included. Selection is a union (so it can
   be fewer than 6 chunks when they overlap), not a fixed concatenation of 6.

Embeddings/TF-IDF are still computed once per unique `content_hash`/`title_hash` across
the combined train+test pool.
""")

code("""
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer

CHUNK_WORDS = CONFIG['chunk_words']
CHUNK_STRIDE = CONFIG['chunk_stride']

def chunk_overlapping(s, chunk_words=CHUNK_WORDS, stride=CHUNK_STRIDE):
    words = s.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [' '.join(words)]
    chunks = []
    for i in range(0, len(words), stride):
        chunk = words[i:i+chunk_words]
        if not chunk:
            break
        chunks.append(' '.join(chunk))
        if i + chunk_words >= len(words):
            break
    return chunks

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
with Timer("retrieval_encode"):
    combined = pd.concat([
        train[['content_hash','title_hash','content_clean','title_clean']],
        test[['content_hash','title_hash','content_clean','title_clean']],
    ], ignore_index=True)

    uniq_content = combined.drop_duplicates('content_hash')[['content_hash','content_clean']].reset_index(drop=True)
    chunk_lists = uniq_content['content_clean'].apply(chunk_overlapping)
    flat_chunks, chunk_ranges = [], {}
    for h, chunks in zip(uniq_content['content_hash'], chunk_lists):
        start = len(flat_chunks)
        flat_chunks.extend(chunks if chunks else [''])
        chunk_ranges[h] = (start, len(flat_chunks))
    print(f"unique articles: {len(uniq_content)}  total overlapping chunks: {len(flat_chunks)} "
          f"(v2 had non-overlapping 35-word chunks; this is expected to be larger)")

    chunk_embs_flat = encode_texts(flat_chunks, prefix='passage: ')

    uniq_title = combined.drop_duplicates('title_hash')[['title_hash','title_clean']].reset_index(drop=True)
    title_embs = encode_texts(uniq_title['title_clean'].tolist(), prefix='query: ', max_length=32)
    title_emb_map = dict(zip(uniq_title['title_hash'], title_embs))
    print(f"chunk embedding matrix: {chunk_embs_flat.shape}  title embedding matrix: {title_embs.shape}")

with Timer("lexical_tfidf_fit"):
    lex_vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=CONFIG['lexical_max_features'])
    lex_vec.fit(pd.concat([uniq_title['title_clean'], pd.Series(flat_chunks)]))
    chunk_tfidf_flat = lex_vec.transform(flat_chunks)
    title_tfidf_all = lex_vec.transform(uniq_title['title_clean'])
    title_tfidf_map = dict(zip(uniq_title['title_hash'], title_tfidf_all))
    print(f"lexical TF-IDF vocab size: {len(lex_vec.vocabulary_)}")
""")

code("""
def retrieve_hybrid(df):
    retrieved_texts, max_sims, n_selected = [], [], []
    for h, th in zip(df['content_hash'], df['title_hash']):
        start, end = chunk_ranges[h]
        c_texts = flat_chunks[start:end]
        n_chunks = len(c_texts)
        sem_sims = chunk_embs_flat[start:end] @ title_emb_map[th]
        lex_sims = np.asarray((chunk_tfidf_flat[start:end] @ title_tfidf_map[th].T).todense()).ravel()

        k_sem = min(CONFIG['top_k_semantic'], n_chunks)
        k_lex = min(CONFIG['top_k_lexical'], n_chunks)
        sem_idx = set(np.argsort(-sem_sims)[:k_sem].tolist())
        lex_idx = set(np.argsort(-lex_sims)[:k_lex].tolist())
        selected = sem_idx | lex_idx
        if CONFIG['include_first_chunk']:
            selected.add(0)
        selected = sorted(selected)

        retrieved_texts.append(' '.join(c_texts[i] for i in selected))
        max_sims.append(float(sem_sims.max()) if n_chunks else 0.0)
        n_selected.append(len(selected))
    return retrieved_texts, np.array(max_sims), np.array(n_selected)

with Timer("retrieval_apply"):
    train['retrieved_text'], train['max_chunk_sim'], train['n_chunks_selected'] = retrieve_hybrid(train)
    test['retrieved_text'], test['max_chunk_sim'], test['n_chunks_selected'] = retrieve_hybrid(test)

print(f"mean chunks selected per article: {train['n_chunks_selected'].mean():.2f} (cap is "
      f"{CONFIG['top_k_semantic']}+{CONFIG['top_k_lexical']}+1 before dedup)")
print(train[['title','retrieved_text','max_chunk_sim']].head(2).to_string())
""")

md("### 6.1 Retrieval coverage diagnostic (EXP2)")

code("""
with Timer("retrieval_coverage_diagnostic"):
    def coverage_row(content_clean, retrieved_text):
        c_ents = extract_caps_entities(content_clean)
        r_ents = extract_caps_entities(retrieved_text)
        ent_cov = safe_div(len(c_ents & r_ents), len(c_ents)) if c_ents else np.nan
        c_nums = set(NUM_RE.findall(content_clean))
        r_nums = set(NUM_RE.findall(retrieved_text))
        num_cov = safe_div(len(c_nums & r_nums), len(c_nums)) if c_nums else np.nan
        return ent_cov, num_cov

    cov = [coverage_row(c, r) for c, r in zip(train['content_clean'], train['retrieved_text'])]
    ent_covs, num_covs = zip(*cov)
    print(f"Median fraction of the FULL article's entities retained by retrieval: {np.nanmedian(ent_covs):.3f}")
    print(f"Median fraction of the FULL article's numbers retained by retrieval:  {np.nanmedian(num_covs):.3f}")
    print("(1.0 = retrieval kept every entity/number that exists anywhere in the article;")
    print(" this is the empirical answer to 'is evidence being thrown away by top-k retrieval'.)")

del chunk_embs_flat, chunk_tfidf_flat  # free memory once baked into the dataframes
""")

# ============================================================
# 7. FEATURE ENGINEERING + CONFLICT TEXT
# ============================================================
md("""
## 7. Feature Engineering + Explicit Conflict Text

Beyond the aggregate relationship features (used later for the fusion diagnostic), we
build a short **conflict-text string** — entities present in the body but absent from the
title, and title numbers absent from the body — and append it to the model's input text.
This makes the entity/number mismatch an explicit textual signal the cross-encoder
attends to, rather than only an aggregate count fed to a separate linear model.
""")

code("""
def build_conflict_text(title, content):
    c_ents = extract_caps_entities(content)
    title_l = title.lower()
    missing_in_title = [e for e in c_ents if e not in title_l][:5]
    tnum = set(NUM_RE.findall(title))
    cnum = set(NUM_RE.findall(content))
    num_conflict = list(tnum - cnum)[:5]
    parts = []
    if missing_in_title:
        parts.append('ENTITAS DI ISI: ' + ', '.join(missing_in_title))
    if num_conflict:
        parts.append('ANGKA JUDUL TAK DITEMUKAN DI ISI: ' + ', '.join(num_conflict))
    return ' | '.join(parts)

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
            'rel_title_len': len(title), 'rel_content_len': len(content),
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

    for df, feats in [(train, train_feats), (test, test_feats)]:
        df['conflict_text'] = [build_conflict_text(t, c) for t, c in zip(df['title_clean'], df['content_clean'])]
        df['model_input_body'] = df['retrieved_text'] + df['conflict_text'].apply(lambda s: (' [SEP] ' + s) if s else '')

print(f"{len(FEATURE_COLS)} relationship features built.")
print(f"rows with a non-empty conflict_text: {(train['conflict_text'].str.len()>0).mean()*100:.1f}%")
print(train[['title','conflict_text']].loc[train['conflict_text'].str.len()>0].head(3).to_string())
""")

# ============================================================
# 8. BASELINE: TF-IDF + LOGISTIC REGRESSION
# ============================================================
md("## 8. Baseline — TF-IDF + Logistic Regression (sanity check)")

code("""
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
""")

# ============================================================
# 9. TRANSFORMER MODEL (IndoBERT cross-encoder) -- PHASE 1
# ============================================================
md("""
## 9. Transformer Model — Phase 1: IndoBERT Cross-Encoder (classification)

Input: `[TITLE] [SEP] [HYBRID-RETRIEVED CHUNKS] [SEP] [CONFLICT TEXT]`. Unchanged from v2
otherwise: class-weighted cross-entropy loss, FP16, early stopping on Macro F1.
""")

code("""
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
    return torch.tensor([n/(2*n0), n/(2*n1)], dtype=torch.float)

def train_cross_encoder(model_name, train_titles, train_bodies, train_labels,
                         val_titles, val_bodies, val_labels, epochs, out_dir, max_len=None,
                         early_stopping=True, fp16_override=None):
    # fp16_override=False is used for DeBERTa-v3 -- its disentangled-attention relative
    # position embeddings are known to be incompatible with some transformers versions'
    # fp16 autocast path (raises "expected scalar type Half but found Float").
    max_len = max_len or CONFIG['max_len']
    use_fp16 = torch.cuda.is_available() if fp16_override is None else (fp16_override and torch.cuda.is_available())
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
            fp16=use_fp16, dataloader_num_workers=0, seed=SEED,
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
            fp16=use_fp16, dataloader_num_workers=0, seed=SEED,
            logging_steps=100, report_to=[], disable_tqdm=False,
        )
        trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds, class_weights=class_weights)
    return trainer, tok
""")

code("""
with Timer("indobert_phase1_training"):
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    indobert_trainer, indobert_tok = train_cross_encoder(
        CONFIG['classifier_model'],
        train['title_clean'].iloc[train_idx], train['model_input_body'].iloc[train_idx], train['label'].iloc[train_idx].values,
        train['title_clean'].iloc[val_idx], train['model_input_body'].iloc[val_idx], train['label'].iloc[val_idx].values,
        epochs=CONFIG['epochs'], out_dir='/kaggle/working/indobert_ckpt',
    )
    indobert_trainer.train()
    indobert_train_time = time.time() - t0
    indobert_gpu_mem = gpu_mem_mb()
print(f"Phase 1 training time: {indobert_train_time:.1f}s   peak GPU mem: {indobert_gpu_mem:.0f} MB")
""")

code("""
with Timer("phase1_validation"):
    val_out = indobert_trainer.predict(indobert_trainer.eval_dataset)
    val_labels = train['label'].iloc[val_idx].values
    val_probs_phase1 = torch.softmax(torch.tensor(val_out.predictions), dim=-1).numpy()[:, 1]
    val_pred_phase1 = (val_probs_phase1 >= 0.5).astype(int)

macro_f1_phase1 = f1_score(val_labels, val_pred_phase1, average='macro')
print(f"Phase 1 Macro F1 (threshold=0.5): {macro_f1_phase1:.4f}")
print(f"F1 class0: {f1_score(val_labels, val_pred_phase1, pos_label=0):.4f}  "
      f"F1 class1: {f1_score(val_labels, val_pred_phase1, pos_label=1):.4f}")
print("Confusion matrix:\\n", confusion_matrix(val_labels, val_pred_phase1))
""")

# ============================================================
# 10. SAME-CONTENT CONTRASTIVE PAIR MINING
# ============================================================
md("""
## 10. Same-Content Contrastive Pair Mining

For every `content_hash` with at least one label=1 title and one label=0 title, every
(positive, negative) title combination becomes a training pair: the model should score
the positive title higher than the negative one **against the same article body**. This
is mined separately for the validation-run's train-side (`train_idx` only, so no val
content leaks in) and for the full dataset (used later for the final retrain).
""")

code("""
def build_contrastive_pairs(df, allowed_hashes):
    pairs = []
    subset = df[df['content_hash'].isin(allowed_hashes)]
    for h, g in subset.groupby('content_hash'):
        pos_rows = g[g.label == 1]
        neg_rows = g[g.label == 0]
        if len(pos_rows) == 0 or len(neg_rows) == 0:
            continue
        for _, prow in pos_rows.iterrows():
            for _, nrow in neg_rows.iterrows():
                pairs.append({
                    'title_pos': prow['title_clean'], 'body_pos': prow['model_input_body'],
                    'title_neg': nrow['title_clean'], 'body_neg': nrow['model_input_body'],
                })
    return pairs

with Timer("contrastive_pair_mining"):
    train_side_hashes = set(train['content_hash'].iloc[train_idx])
    contrastive_pairs_val_run = build_contrastive_pairs(train, train_side_hashes)
    contrastive_pairs_full = build_contrastive_pairs(train, set(train['content_hash']))
    print(f"contrastive pairs for the validation run's Phase 2 (train_idx only): {len(contrastive_pairs_val_run)}")
    print(f"contrastive pairs for the final full-data retrain's Phase 2: {len(contrastive_pairs_full)}")
    if contrastive_pairs_val_run:
        ex = contrastive_pairs_val_run[0]
        print(f"\\nExample pair:\\n  POS title: {ex['title_pos']}\\n  NEG title: {ex['title_neg']}")
""")

# ============================================================
# 11. PHASE 2: SAME-CONTENT CONTRASTIVE FINE-TUNING
# ============================================================
md("""
## 11. Phase 2 — Same-Content Contrastive Fine-Tuning

Continues training the **same** Phase-1 model weights with a margin ranking loss:
`score(pos) > score(neg) + margin` for same-content pairs. This is a small, fast extra
training stage (a few hundred/thousand pairs, 2 epochs) — its purpose is narrow and
targeted: sharpen exactly the "same topic, different fact" boundary that Phase 1's
generic classification objective does not explicitly enforce.
""")

code("""
class ContrastivePairDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len):
        self.pairs = pairs; self.tok = tokenizer; self.max_len = max_len
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        p = self.pairs[idx]
        enc_pos = self.tok(p['title_pos'], p['body_pos'], truncation=True, max_length=self.max_len,
                            padding='max_length', return_tensors='pt')
        enc_neg = self.tok(p['title_neg'], p['body_neg'], truncation=True, max_length=self.max_len,
                            padding='max_length', return_tensors='pt')
        return {
            'pos_input_ids': enc_pos['input_ids'].squeeze(0), 'pos_attention_mask': enc_pos['attention_mask'].squeeze(0),
            'neg_input_ids': enc_neg['input_ids'].squeeze(0), 'neg_attention_mask': enc_neg['attention_mask'].squeeze(0),
        }

def contrastive_finetune(model, tokenizer, pairs, epochs, lr, margin, batch_size, max_len, device, label=''):
    if not pairs:
        print(f"[{label}] no same-content contrastive pairs available -- skipping Phase 2.")
        return model, {'pairs': 0, 'loss_history': []}
    ds = ContrastivePairDataset(pairs, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MarginRankingLoss(margin=margin)
    model.to(device); model.train()
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pos_logits = model(input_ids=batch['pos_input_ids'], attention_mask=batch['pos_attention_mask']).logits
            neg_logits = model(input_ids=batch['neg_input_ids'], attention_mask=batch['neg_attention_mask']).logits
            pos_score = torch.softmax(pos_logits, dim=-1)[:, 1]
            neg_score = torch.softmax(neg_logits, dim=-1)[:, 1]
            target = torch.ones_like(pos_score)
            loss = loss_fn(pos_score, neg_score, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(loader), 1)
        history.append(avg_loss)
        print(f"[{label}] contrastive epoch {epoch+1}/{epochs} loss={avg_loss:.4f}")
    model.eval()
    return model, {'pairs': len(pairs), 'loss_history': history}
""")

code("""
with Timer("phase2_contrastive_finetune"):
    indobert_trainer.model, phase2_info = contrastive_finetune(
        indobert_trainer.model, indobert_tok, contrastive_pairs_val_run,
        epochs=CONFIG['contrastive_epochs'], lr=CONFIG['contrastive_lr'], margin=CONFIG['contrastive_margin'],
        batch_size=CONFIG['contrastive_batch_size'], max_len=CONFIG['max_len'], device=DEVICE, label='validation-run',
    )

with Timer("phase2_validation"):
    val_out2 = indobert_trainer.predict(indobert_trainer.eval_dataset)
    val_probs_phase2 = torch.softmax(torch.tensor(val_out2.predictions), dim=-1).numpy()[:, 1]
    val_pred_phase2 = (val_probs_phase2 >= 0.5).astype(int)

macro_f1_phase2 = f1_score(val_labels, val_pred_phase2, average='macro') if phase2_info['pairs'] > 0 else -1
print(f"Phase 1 Macro F1 @0.5: {macro_f1_phase1:.4f}")
print(f"Phase 1+2 Macro F1 @0.5: {macro_f1_phase2:.4f}" if phase2_info['pairs']>0 else "Phase 2 skipped (no pairs).")
if phase2_info['pairs'] > 0:
    print("Confusion matrix (Phase 1+2):\\n", confusion_matrix(val_labels, val_pred_phase2))

USE_PHASE2 = phase2_info['pairs'] > 0 and macro_f1_phase2 >= macro_f1_phase1
val_probs = val_probs_phase2 if USE_PHASE2 else val_probs_phase1
print(f"\\n-> {'Contrastive fine-tuning HELPED' if USE_PHASE2 else 'Contrastive fine-tuning did NOT help on this split -- falling back to Phase 1 only'} "
      f"(this decision is reused for the final full-data model in Section 17).")
""")

# ============================================================
# 12. THRESHOLD TUNING
# ============================================================
md("## 12. Threshold Tuning")

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
print(f"\\nBest threshold: {BEST_THRESHOLD}  (Macro F1={BEST_VAL_MACRO_F1:.4f})")
""")

# ============================================================
# 13. VALIDATION DIAGNOSIS (EXP1): COLD-START VS COMPETITION-REALISTIC
# ============================================================
md("""
## 13. Validation Diagnosis (EXP1) — Cold-Start vs Competition-Realistic Estimate

`GroupShuffleSplit` deliberately keeps val_idx's content bodies unseen during training —
a genuine "new article" (cold-start) estimate. The real test set is NOT cold in the same
way: Section 3.1 found **{TEST_OVERRIDE_RATE:.1%} of test rows have an exact
(title,content) match in train**, which get a free correct answer at submission time
(Section 18). This simulates what the blended score should look like, without needing
real test labels.
""")

code("""
with Timer("validation_diagnosis_exp1"):
    rng = np.random.RandomState(SEED)
    n_sim_override = int(round(TEST_OVERRIDE_RATE * len(val_labels)))
    sim_scores = []
    val_pred_best = (val_probs >= BEST_THRESHOLD).astype(int)
    for _ in range(30):
        sim_pred = val_pred_best.copy()
        override_idx = rng.choice(len(sim_pred), size=n_sim_override, replace=False)
        sim_pred[override_idx] = val_labels[override_idx]
        sim_scores.append(f1_score(val_labels, sim_pred, average='macro'))

print(f"Cold-start Macro F1 (genuinely new articles): {BEST_VAL_MACRO_F1:.4f}")
print(f"Simulated competition-realistic Macro F1 (assuming {TEST_OVERRIDE_RATE*100:.1f}% of rows get a perfect "
      f"lookup, matching test's real overlap rate, mean of 30 simulations): "
      f"{np.mean(sim_scores):.4f} +/- {np.std(sim_scores):.4f}")
print("NOTE: this assumes perfect lookup for a random subset at the observed rate -- a plausible")
print("  upper-context band for what the leaderboard should look like, not a guarantee.")
""")

# ============================================================
# 14. HARD NEGATIVE ANALYSIS
# ============================================================
md("## 14. Hard Negative Analysis")

code("""
with Timer("hard_negative_analysis"):
    val_df = train.iloc[val_idx].copy()
    val_df['pred'] = val_pred_best
    val_df['prob'] = val_probs
    val_df = val_df.join(train_feats.iloc[val_idx])

    mistakes = val_df[val_df['label'] != val_df['pred']].copy()
    high_sim_wrong = mistakes[(mistakes['label']==0) & (mistakes['max_chunk_sim'] > mistakes['max_chunk_sim'].median())]
    print(f"total validation mistakes: {len(mistakes)} / {len(val_df)}")
    print(f"high-similarity false positives: {len(high_sim_wrong)}")

    pd.set_option('display.max_colwidth', 140)
    cols = ['title','retrieved_text','label','pred','prob','max_chunk_sim','number_mismatch','date_overlap','entity_coverage_in_title','negation_mismatch']
    display_cols = [c for c in cols if c in high_sim_wrong.columns]
    print("\\n=== Representative hard negatives (high similarity, wrongly predicted) ===")
    print(high_sim_wrong[display_cols].head(15).to_string())
""")

# ============================================================
# 15. OPTIONAL FEATURE FUSION
# ============================================================
md("## 15. Optional Feature Fusion")

code("""
FUSION_IMPROVEMENT = 0.0
if CONFIG['run_feature_fusion']:
    with Timer("feature_fusion"):
        from sklearn.model_selection import train_test_split as _tts
        from sklearn.preprocessing import StandardScaler
        fz_idx = np.arange(len(val_df))
        fz_tr, fz_te = _tts(fz_idx, test_size=0.4, random_state=SEED, stratify=val_df['label'])

        X_indobert_only = val_df[['prob']].values
        X_fused = pd.concat([val_df[['prob']], val_df[FEATURE_COLS]], axis=1).values
        y_fz = val_df['label'].values

        f1_only, f1_fused = [], []
        for X, store in [(X_indobert_only, f1_only), (X_fused, f1_fused)]:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X[fz_tr]); X_te_s = scaler.transform(X[fz_te])
            clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED)
            clf.fit(X_tr_s, y_fz[fz_tr])
            store.append(f1_score(y_fz[fz_te], clf.predict(X_te_s), average='macro'))

        print(f"IndoBERT-prob-only Macro F1 (diagnostic split): {f1_only[0]:.4f}")
        print(f"IndoBERT-prob + relationship features Macro F1 (diagnostic split): {f1_fused[0]:.4f}")
        FUSION_IMPROVEMENT = f1_fused[0] - f1_only[0]
        print(f"-> {'Worth keeping' if FUSION_IMPROVEMENT>=0.01 else 'Negligible, keep the simpler model'} "
              f"({FUSION_IMPROVEMENT:+.4f}).")
else:
    print("Feature fusion diagnostic skipped.")
""")

# ============================================================
# 16. OPTIONAL DEBERTA BENCHMARK (fp16 bug fixed)
# ============================================================
md("""
## 16. Optional DeBERTa-v3-small Benchmark (FP16 bug fixed)

v2's DeBERTa run crashed with `expected scalar type Half but found Float` — a known
DeBERTa-v2/v3 disentangled-attention incompatibility with fp16 autocast in some
transformers versions. Fixed here by disabling fp16 specifically for this model
(`fp16_override=False`); IndoBERT is unaffected and keeps fp16.
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
                train['title_clean'].iloc[train_idx], train['model_input_body'].iloc[train_idx], train['label'].iloc[train_idx].values,
                train['title_clean'].iloc[val_idx], train['model_input_body'].iloc[val_idx], train['label'].iloc[val_idx].values,
                epochs=CONFIG['deberta_epochs'], out_dir='/kaggle/working/deberta_ckpt',
                fp16_override=False,
            )
            deberta_trainer.train()
            deberta_train_time = time.time() - t0
            deberta_gpu_mem = gpu_mem_mb()

            t0 = time.time()
            d_out = deberta_trainer.predict(deberta_trainer.eval_dataset)
            deberta_infer_time = time.time() - t0
            d_probs = torch.softmax(torch.tensor(d_out.predictions), dim=-1).numpy()[:, 1]
            deberta_macro_f1 = f1_score(val_labels, (d_probs >= 0.5).astype(int), average='macro')

        deberta_results = {'model': CONFIG['alt_model'], 'val_macro_f1': deberta_macro_f1,
                            'train_time_s': deberta_train_time, 'infer_time_s': deberta_infer_time,
                            'gpu_mem_mb': deberta_gpu_mem}
        print(deberta_results)
    except Exception as e:
        print(f"[warn] DeBERTa benchmark failed/skipped: {e}")
else:
    print("DeBERTa benchmark skipped.")
""")

code("""
with Timer("indobert_infer_timing"):
    t0 = time.time()
    _ = indobert_trainer.predict(indobert_trainer.eval_dataset)
    indobert_infer_time = time.time() - t0

compare_rows = [{'model': CONFIG['classifier_model'] + (' +contrastive' if USE_PHASE2 else ''),
                  'val_macro_f1': BEST_VAL_MACRO_F1, 'train_time_s': indobert_train_time,
                  'infer_time_s': indobert_infer_time, 'gpu_mem_mb': indobert_gpu_mem}]
if deberta_results is not None:
    compare_rows.append(deberta_results)
compare_df = pd.DataFrame(compare_rows)
print(compare_df.to_string(index=False))

if deberta_results is not None and deberta_results['val_macro_f1'] > BEST_VAL_MACRO_F1 + 0.01 and deberta_results['train_time_s'] < indobert_train_time * 2:
    BEST_ARCHITECTURE = CONFIG['alt_model']
    print(f"\\n-> Selecting {BEST_ARCHITECTURE}: meaningfully better without disproportionate cost.")
else:
    BEST_ARCHITECTURE = CONFIG['classifier_model']
    print(f"\\n-> Keeping {BEST_ARCHITECTURE} (+ contrastive phase if it helped): best trade-off.")
""")

# ============================================================
# 17. FINAL TRAINING (Phase 1 + Phase 2 on full data)
# ============================================================
md("""
## 17. Final Training — Phase 1 + Phase 2 on Full Data

Retrains on all of `train`, reusing the architecture and the **same Phase-2 decision**
(`USE_PHASE2`) validated above — there's no held-out set left to re-decide on, so we
apply what was empirically shown to work (or not) during validation.
""")

code("""
with Timer("final_training"):
    t0 = time.time()
    final_epochs = max(1, int(round(indobert_trainer.state.epoch or CONFIG['epochs'])))
    print(f"Final Phase 1 epochs (matched to validated run): {final_epochs}")

    all_labels = train['label'].values
    final_trainer, final_tok = train_cross_encoder(
        BEST_ARCHITECTURE,
        train['title_clean'], train['model_input_body'], all_labels,
        None, None, None,
        epochs=final_epochs, out_dir='/kaggle/working/final_ckpt',
        early_stopping=False,
    )
    final_trainer.train()

    if USE_PHASE2 and BEST_ARCHITECTURE == CONFIG['classifier_model']:
        final_trainer.model, final_phase2_info = contrastive_finetune(
            final_trainer.model, final_tok, contrastive_pairs_full,
            epochs=CONFIG['contrastive_epochs'], lr=CONFIG['contrastive_lr'], margin=CONFIG['contrastive_margin'],
            batch_size=CONFIG['contrastive_batch_size'], max_len=CONFIG['max_len'], device=DEVICE, label='final-retrain',
        )
    else:
        print("Skipping final Phase 2 (either it didn't help validation, or DeBERTa was selected).")
    final_train_time = time.time() - t0
print(f"Final model trained on {len(train)} rows in {final_train_time:.1f}s")
""")

# ============================================================
# 18. TEST PREDICTION
# ============================================================
md("## 18. Test Prediction")

code("""
with Timer("test_inference"):
    t0 = time.time()
    test_ds = PairDataset(test['title_clean'], test['model_input_body'], None, final_tok, CONFIG['max_len'])
    test_out = final_trainer.predict(test_ds)
    test_probs = torch.softmax(torch.tensor(test_out.predictions), dim=-1).numpy()[:, 1]
    test_pred = (test_probs >= BEST_THRESHOLD).astype(int)
    final_infer_time = time.time() - t0
print(f"Test inference done in {final_infer_time:.1f}s. Positive-rate: {test_pred.mean():.3f}")
""")

# ============================================================
# 19. SUBMISSION
# ============================================================
md("## 19. Submission (with exact-match override)")

code("""
with Timer("submission"):
    final_pred = test_pred.copy()
    for i, key in enumerate(test_override_keys):
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
print(f"\\nExact-match override applied to {n_overridden_test} test rows ({TEST_OVERRIDE_RATE*100:.1f}%).")
""")

# ============================================================
# FINAL OUTPUT
# ============================================================
md("## Final Runtime and Results Summary")

code("""
TOTAL_RUNTIME = time.time() - RUN_T0
print("=" * 55)
print("Best Model:", BEST_ARCHITECTURE, "(+contrastive)" if USE_PHASE2 else "(Phase 1 only)")
print(f"Best Validation Macro F1 (cold-start): {BEST_VAL_MACRO_F1:.4f}")
print(f"Simulated competition-realistic Macro F1: {np.mean(sim_scores):.4f}")
print(f"Best Threshold: {BEST_THRESHOLD}")
print(f"Approximate GPU Memory: {max(indobert_gpu_mem, (deberta_results or {}).get('gpu_mem_mb', 0)):.0f} MB")
print(f"Submission Shape: {submission.shape}")
print("Submission Path: /kaggle/working/submission.csv")
print("=" * 55)

print(f"\\nPhase 1 -> Phase 1+2 Macro F1: {macro_f1_phase1:.4f} -> "
      f"{macro_f1_phase2:.4f}" if phase2_info['pairs']>0 else "\\nPhase 2 was skipped (no pairs).")
print(f"Baseline (TF-IDF+LogReg) Macro F1 was {baseline_macro_f1:.4f}")
print(f"Feature-fusion improvement (diagnostic): {FUSION_IMPROVEMENT:+.4f}")
print(f"\\nTotal notebook runtime: {TOTAL_RUNTIME:.1f}s ({TOTAL_RUNTIME/60:.1f} min)")
print("\\nPer-section timings:")
for k, v in sorted(TIMINGS.items(), key=lambda kv: -kv[1]):
    print(f"  {k}: {v:.1f}s")
""")

nb['cells'] = cells
nb['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb['metadata']['language_info'] = {'name': 'python', 'version': '3.11'}

with open('ifest2026_dac_v3_contrastive.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook written.")
