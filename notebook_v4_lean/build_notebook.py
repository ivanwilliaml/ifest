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
# Penyisihan IFEST 2026 DAC — v4: Lean Submission-Only Pipeline

Same modeling ideas as v3 (hybrid overlapping retrieval + same-content contrastive
fine-tuning), but **every step whose output does not feed the final submission has been
removed** to cut runtime, per explicit instruction. Compared to v3, this notebook drops:

- TF-IDF baseline (was comparison-only)
- Hard-negative analysis (was interpretability-only)
- Feature-fusion diagnostic (was never deployed to the final model)
- DeBERTa-v3-small benchmark (the single biggest removed cost, ~15-20 min; rarely won
  anyway, and IndoBERT is now hardcoded as the architecture)
- EXP1 cold-start-vs-realistic simulation and EXP2 retrieval-coverage diagnostic
- The aggregate relationship-feature table (`build_relationship_features`/`FEATURE_COLS`)
  — it only existed to feed the two removed diagnostics

**Trade-off, stated plainly:** if this run's score is still unsatisfying, this notebook
gives no diagnostic trail to explain why (no confusion-matrix breakdown by cause, no hard
negative examples, no retrieval-coverage numbers) — that visibility was intentionally
traded for a faster, submission-only run. Re-add v3's diagnostics if root-causing is
needed again.

**What's kept because its output feeds the submission:** hybrid retrieval, the
conflict-text builder (goes into the model's input text), Phase 1 classification
fine-tuning, Phase 2 same-content contrastive fine-tuning (with its Phase-1-vs-Phase-1+2
comparison — needed to decide whether Phase 2 is used at all), threshold tuning (produces
`BEST_THRESHOLD`), and the exact-match override.
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
    return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
""")

# ============================================================
# 2. CONFIGURATION
# ============================================================
md("## 2. Configuration")

code("""
CONFIG = {
    'id_col': 'id', 'title_col': 'title', 'content_col': 'content', 'label_col': 'label',
    'seed': SEED,

    'retriever_model': 'intfloat/multilingual-e5-small',
    'chunk_words': 50,
    'chunk_stride': 25,
    'top_k_semantic': 3,
    'top_k_lexical': 2,
    'include_first_chunk': True,
    'retriever_max_len': 96,
    'retriever_batch_size': 256,
    'lexical_max_features': 60000,

    'classifier_model': 'indobenchmark/indobert-base-p1',
    'max_len': 384,
    'batch_size': 16,
    'lr': 2e-5,
    'epochs': 3,
    'early_stopping_patience': 1,
    'val_size': 0.15,

    'contrastive_epochs': 2,
    'contrastive_lr': 1e-5,
    'contrastive_margin': 0.15,
    'contrastive_batch_size': 8,

    'threshold_grid': [round(x, 2) for x in np.arange(0.20, 0.81, 0.05)],
}
print(json.dumps(CONFIG, indent=2, default=str))
""")

# ============================================================
# 3. LOAD DATA
# ============================================================
md("## 3. Load Data")

code("""
with Timer("data_loading"):
    DATA_FILES = {}
    for root, dirs, files in os.walk('/kaggle/input'):
        for f in files:
            if f in ('train.csv', 'test.csv', 'sample_submission.csv'):
                DATA_FILES.setdefault(f, os.path.join(root, f))
    assert len(DATA_FILES) == 3, f"Missing data files: {DATA_FILES}"
    train = pd.read_csv(DATA_FILES['train.csv'])
    test = pd.read_csv(DATA_FILES['test.csv'])
    sample_sub = pd.read_csv(DATA_FILES['sample_submission.csv'])

assert list(train.columns) == ['id','title','content','label']
assert list(test.columns) == ['id','title','content']
print("train shape:", train.shape, " test shape:", test.shape)

def normalize_for_hash(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    s = re.sub(r'\\s+', ' ', s).strip()
    return s

for df in (train, test):
    df['content_hash'] = df['content'].apply(lambda s: hashlib.md5(normalize_for_hash(s).encode()).hexdigest())
    df['title_hash'] = df['title'].apply(lambda s: hashlib.md5(normalize_for_hash(s).encode()).hexdigest())

with Timer("exact_pair_overlap_check"):
    exact_pair_map = (train.drop_duplicates(subset=['title_hash','content_hash'])
                            .set_index(['title_hash','content_hash'])['label'].to_dict())
    test_override_keys = list(zip(test['title_hash'], test['content_hash']))
    n_overridden_test = sum(1 for k in test_override_keys if k in exact_pair_map)
    print(f"test rows with an exact (title,content) match in train: {n_overridden_test}/{len(test)} "
          f"({n_overridden_test/len(test)*100:.1f}%) -- used for the Section 15 submission override.")
""")

# ============================================================
# 4. LEAKAGE-SAFE SPLIT
# ============================================================
md("## 4. Leakage-Safe Split (GroupShuffleSplit on content_hash)")

code("""
from sklearn.model_selection import GroupShuffleSplit

with Timer("split"):
    gss = GroupShuffleSplit(n_splits=1, test_size=CONFIG['val_size'], random_state=SEED)
    train_idx, val_idx = next(gss.split(train, train['label'], groups=train['content_hash']))
    overlap = set(train['content_hash'].iloc[train_idx]) & set(train['content_hash'].iloc[val_idx])
    assert len(overlap) == 0
    print(f"train rows: {len(train_idx)}   val rows: {len(val_idx)}   (zero content_hash overlap confirmed)")
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
    return WS_RE.sub(' ', s).strip()

with Timer("text_cleaning"):
    for df in (train, test):
        df['title_clean'] = df['title'].apply(clean_text)
        df['content_clean'] = df['content'].apply(clean_text)
""")

# ============================================================
# 6. HYBRID RETRIEVAL + CONFLICT TEXT
# ============================================================
md("""
## 6. Hybrid Overlapping Retrieval + Conflict Text

Overlapping 50-word/stride-25 chunks; top-3 E5-semantic + top-2 TF-IDF-lexical + lede
chunk, unioned. A short conflict-text string (entities/numbers in the body but absent
from the title) is appended to the model's input text.
""")

code("""
NUM_RE = re.compile(r'\\d+[.,]?\\d*\\s*%?')

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

def build_conflict_text(title, content):
    c_ents = extract_caps_entities(content)
    title_l = title.lower()
    missing_in_title = [e for e in c_ents if e not in title_l][:5]
    tnum = set(NUM_RE.findall(title)); cnum = set(NUM_RE.findall(content))
    num_conflict = list(tnum - cnum)[:5]
    parts = []
    if missing_in_title: parts.append('ENTITAS DI ISI: ' + ', '.join(missing_in_title))
    if num_conflict: parts.append('ANGKA JUDUL TAK DITEMUKAN DI ISI: ' + ', '.join(num_conflict))
    return ' | '.join(parts)
""")

code("""
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer

CHUNK_WORDS, CHUNK_STRIDE = CONFIG['chunk_words'], CONFIG['chunk_stride']

def chunk_overlapping(s, chunk_words=CHUNK_WORDS, stride=CHUNK_STRIDE):
    words = s.split()
    if not words: return []
    if len(words) <= chunk_words: return [' '.join(words)]
    chunks = []
    for i in range(0, len(words), stride):
        chunk = words[i:i+chunk_words]
        if not chunk: break
        chunks.append(' '.join(chunk))
        if i + chunk_words >= len(words): break
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
    print(f"unique articles: {len(uniq_content)}  total overlapping chunks: {len(flat_chunks)}")

    chunk_embs_flat = encode_texts(flat_chunks, prefix='passage: ')

    uniq_title = combined.drop_duplicates('title_hash')[['title_hash','title_clean']].reset_index(drop=True)
    title_embs = encode_texts(uniq_title['title_clean'].tolist(), prefix='query: ', max_length=32)
    title_emb_map = dict(zip(uniq_title['title_hash'], title_embs))

with Timer("lexical_tfidf_fit"):
    lex_vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=CONFIG['lexical_max_features'])
    lex_vec.fit(pd.concat([uniq_title['title_clean'], pd.Series(flat_chunks)]))
    chunk_tfidf_flat = lex_vec.transform(flat_chunks)
    title_tfidf_map = dict(zip(uniq_title['title_hash'], lex_vec.transform(uniq_title['title_clean'])))
""")

code("""
def retrieve_hybrid(df):
    retrieved_texts = []
    for h, th in zip(df['content_hash'], df['title_hash']):
        start, end = chunk_ranges[h]
        c_texts = flat_chunks[start:end]
        n_chunks = len(c_texts)
        sem_sims = chunk_embs_flat[start:end] @ title_emb_map[th]
        lex_sims = np.asarray((chunk_tfidf_flat[start:end] @ title_tfidf_map[th].T).todense()).ravel()
        k_sem = min(CONFIG['top_k_semantic'], n_chunks)
        k_lex = min(CONFIG['top_k_lexical'], n_chunks)
        selected = set(np.argsort(-sem_sims)[:k_sem].tolist()) | set(np.argsort(-lex_sims)[:k_lex].tolist())
        if CONFIG['include_first_chunk']:
            selected.add(0)
        retrieved_texts.append(' '.join(c_texts[i] for i in sorted(selected)))
    return retrieved_texts

with Timer("retrieval_apply"):
    train['retrieved_text'] = retrieve_hybrid(train)
    test['retrieved_text'] = retrieve_hybrid(test)

    for df in (train, test):
        df['conflict_text'] = [build_conflict_text(t, c) for t, c in zip(df['title_clean'], df['content_clean'])]
        df['model_input_body'] = df['retrieved_text'] + df['conflict_text'].apply(lambda s: (' [SEP] ' + s) if s else '')

del chunk_embs_flat, chunk_tfidf_flat
print(train[['title','model_input_body']].head(2).to_string())
""")

# ============================================================
# 7. PHASE 1: INDOBERT CLASSIFICATION
# ============================================================
md("""
## 7. Phase 1 — IndoBERT Cross-Encoder Fine-Tune

Class-weighted cross-entropy, FP16, early stopping on Macro F1.
""")

code("""
from transformers import (AutoModelForSequenceClassification, Trainer, TrainingArguments,
                           EarlyStoppingCallback)
from sklearn.metrics import f1_score, confusion_matrix

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
    return {'macro_f1': f1_score(labels, preds, average='macro')}

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

def make_class_weights(labels):
    n = len(labels); n0 = (labels==0).sum(); n1 = (labels==1).sum()
    return torch.tensor([n/(2*n0), n/(2*n1)], dtype=torch.float)

def train_cross_encoder(model_name, train_titles, train_bodies, train_labels,
                         val_titles, val_bodies, val_labels, epochs, out_dir,
                         max_len=None, early_stopping=True):
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

val_out = indobert_trainer.predict(indobert_trainer.eval_dataset)
val_labels = train['label'].iloc[val_idx].values
val_probs_phase1 = torch.softmax(torch.tensor(val_out.predictions), dim=-1).numpy()[:, 1]
macro_f1_phase1 = f1_score(val_labels, (val_probs_phase1>=0.5).astype(int), average='macro')
print(f"Phase 1 training time: {indobert_train_time:.1f}s   Macro F1 @0.5: {macro_f1_phase1:.4f}")
""")

# ============================================================
# 8. SAME-CONTENT CONTRASTIVE PAIR MINING
# ============================================================
md("## 8. Same-Content Contrastive Pair Mining")

code("""
def build_contrastive_pairs(df, allowed_hashes):
    pairs = []
    subset = df[df['content_hash'].isin(allowed_hashes)]
    for h, g in subset.groupby('content_hash'):
        pos_rows = g[g.label == 1]; neg_rows = g[g.label == 0]
        if len(pos_rows) == 0 or len(neg_rows) == 0:
            continue
        for _, prow in pos_rows.iterrows():
            for _, nrow in neg_rows.iterrows():
                pairs.append({'title_pos': prow['title_clean'], 'body_pos': prow['model_input_body'],
                              'title_neg': nrow['title_clean'], 'body_neg': nrow['model_input_body']})
    return pairs

with Timer("contrastive_pair_mining"):
    contrastive_pairs_val_run = build_contrastive_pairs(train, set(train['content_hash'].iloc[train_idx]))
    contrastive_pairs_full = build_contrastive_pairs(train, set(train['content_hash']))
    print(f"pairs for validation-run Phase 2: {len(contrastive_pairs_val_run)}   "
          f"pairs for final-retrain Phase 2: {len(contrastive_pairs_full)}")
""")

# ============================================================
# 9. PHASE 2: CONTRASTIVE FINE-TUNING
# ============================================================
md("""
## 9. Phase 2 — Same-Content Contrastive Fine-Tuning

Margin ranking loss on same-content pairs, continuing from Phase-1 weights. Result is
compared against Phase 1 alone; whichever validates better is used (see Section 10).
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
        return {'pos_input_ids': enc_pos['input_ids'].squeeze(0), 'pos_attention_mask': enc_pos['attention_mask'].squeeze(0),
                'neg_input_ids': enc_neg['input_ids'].squeeze(0), 'neg_attention_mask': enc_neg['attention_mask'].squeeze(0)}

def contrastive_finetune(model, tokenizer, pairs, epochs, lr, margin, batch_size, max_len, device, label=''):
    if not pairs:
        print(f"[{label}] no pairs -- skipping Phase 2.")
        return model, {'pairs': 0}
    ds = ContrastivePairDataset(pairs, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MarginRankingLoss(margin=margin)
    model.to(device); model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pos_logits = model(input_ids=batch['pos_input_ids'], attention_mask=batch['pos_attention_mask']).logits
            neg_logits = model(input_ids=batch['neg_input_ids'], attention_mask=batch['neg_attention_mask']).logits
            pos_score = torch.softmax(pos_logits, dim=-1)[:, 1]
            neg_score = torch.softmax(neg_logits, dim=-1)[:, 1]
            loss = loss_fn(pos_score, neg_score, torch.ones_like(pos_score))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"[{label}] contrastive epoch {epoch+1}/{epochs} loss={total_loss/max(len(loader),1):.4f}")
    model.eval()
    return model, {'pairs': len(pairs)}

with Timer("phase2_contrastive_finetune"):
    indobert_trainer.model, phase2_info = contrastive_finetune(
        indobert_trainer.model, indobert_tok, contrastive_pairs_val_run,
        epochs=CONFIG['contrastive_epochs'], lr=CONFIG['contrastive_lr'], margin=CONFIG['contrastive_margin'],
        batch_size=CONFIG['contrastive_batch_size'], max_len=CONFIG['max_len'], device=DEVICE, label='validation-run',
    )

val_out2 = indobert_trainer.predict(indobert_trainer.eval_dataset)
val_probs_phase2 = torch.softmax(torch.tensor(val_out2.predictions), dim=-1).numpy()[:, 1]
macro_f1_phase2 = f1_score(val_labels, (val_probs_phase2>=0.5).astype(int), average='macro') if phase2_info['pairs']>0 else -1
print(f"Phase 1+2 Macro F1 @0.5: {macro_f1_phase2:.4f}" if phase2_info['pairs']>0 else "Phase 2 skipped.")
""")

# ============================================================
# 10. PHASE 2 KEEP/DISCARD DECISION
# ============================================================
md("## 10. Phase 2 Keep/Discard Decision")

code("""
USE_PHASE2 = phase2_info['pairs'] > 0 and macro_f1_phase2 >= macro_f1_phase1
val_probs = val_probs_phase2 if USE_PHASE2 else val_probs_phase1
print(f"Phase 1: {macro_f1_phase1:.4f}   Phase 1+2: {macro_f1_phase2:.4f}" if phase2_info['pairs']>0 else f"Phase 1 only: {macro_f1_phase1:.4f}")
print(f"-> USE_PHASE2 = {USE_PHASE2} (this decision is reused for the final full-data model).")
""")

# ============================================================
# 11. THRESHOLD TUNING
# ============================================================
md("## 11. Threshold Tuning")

code("""
with Timer("threshold_tuning"):
    thr_rows = []
    for thr in CONFIG['threshold_grid']:
        pred = (val_probs >= thr).astype(int)
        thr_rows.append({'threshold': thr, 'macro_f1': f1_score(val_labels, pred, average='macro')})
    thr_df = pd.DataFrame(thr_rows)
    BEST_THRESHOLD = float(thr_df.loc[thr_df['macro_f1'].idxmax(), 'threshold'])
    BEST_VAL_MACRO_F1 = float(thr_df['macro_f1'].max())
print(thr_df.to_string(index=False))
print(f"\\nBest threshold: {BEST_THRESHOLD}  (Macro F1={BEST_VAL_MACRO_F1:.4f})")
""")

# ============================================================
# 12. FINAL TRAINING (full data)
# ============================================================
md("## 12. Final Training — Phase 1 + Phase 2 on Full Data")

code("""
with Timer("final_training"):
    t0 = time.time()
    final_epochs = max(1, int(round(indobert_trainer.state.epoch or CONFIG['epochs'])))
    all_labels = train['label'].values
    final_trainer, final_tok = train_cross_encoder(
        CONFIG['classifier_model'],
        train['title_clean'], train['model_input_body'], all_labels,
        None, None, None,
        epochs=final_epochs, out_dir='/kaggle/working/final_ckpt', early_stopping=False,
    )
    final_trainer.train()

    if USE_PHASE2:
        final_trainer.model, _ = contrastive_finetune(
            final_trainer.model, final_tok, contrastive_pairs_full,
            epochs=CONFIG['contrastive_epochs'], lr=CONFIG['contrastive_lr'], margin=CONFIG['contrastive_margin'],
            batch_size=CONFIG['contrastive_batch_size'], max_len=CONFIG['max_len'], device=DEVICE, label='final-retrain',
        )
    final_train_time = time.time() - t0
print(f"Final model trained on {len(train)} rows in {final_train_time:.1f}s (epochs={final_epochs}, phase2={USE_PHASE2})")
""")

# ============================================================
# 13. TEST INFERENCE
# ============================================================
md("## 13. Test Inference")

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
# 14. SUBMISSION
# ============================================================
md("## 14. Submission (with exact-match override)")

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
print(f"Exact-match override applied to {n_overridden_test} test rows ({n_overridden_test/len(test)*100:.1f}%).")
""")

# ============================================================
# FINAL SUMMARY
# ============================================================
md("## Final Summary")

code("""
TOTAL_RUNTIME = time.time() - RUN_T0
print("=" * 50)
print(f"Architecture: {CONFIG['classifier_model']} {'+contrastive' if USE_PHASE2 else '(Phase 1 only)'}")
print(f"Validation Macro F1: {BEST_VAL_MACRO_F1:.4f}   Best threshold: {BEST_THRESHOLD}")
print(f"Peak GPU memory: {indobert_gpu_mem:.0f} MB")
print(f"Submission: {submission.shape} -> /kaggle/working/submission.csv")
print(f"Total runtime: {TOTAL_RUNTIME:.1f}s ({TOTAL_RUNTIME/60:.1f} min)")
print("=" * 50)
print("\\nPer-section timings:")
for k, v in sorted(TIMINGS.items(), key=lambda kv: -kv[1]):
    print(f"  {k}: {v:.1f}s")
""")

nb['cells'] = cells
nb['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb['metadata']['language_info'] = {'name': 'python', 'version': '3.11'}

with open('ifest2026_dac_v4_lean.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook written.")
