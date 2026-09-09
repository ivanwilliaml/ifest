import nbformat as nbf
import textwrap

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(src).strip()))

def code(src):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(src).strip()))

# ============================================================
# 1. COMPETITION OBJECTIVE
# ============================================================
md("""
# Penyisihan IFEST 2026 DAC — Headline/Body Consistency: Fast Leakage-Safe Baseline

## 1. Competition Objective

Given `(title, content)` pairs, predict whether the headline's main factual claim is
**supported** (`label=1`) or **inconsistent / misleading / different-event** (`label=0`)
relative to the article body. Metric: **Macro F1**.

**Framing.** This is a *headline–body relationship* problem, not ordinary document
classification (cf. Fake News Challenge stance-detection literature). High lexical
similarity does **not** imply consistency — e.g. a title claiming "500 ribu kasus" against
a body reporting "300 ribu kasus" is topically identical but factually inconsistent. The
features below are chosen to probe topical, lexical, numeric and negation-level agreement
between the two texts, not just raw similarity.

**Engineering constraint.** Strongest possible baseline at very low runtime: target
< 10 min, hard budget < 20 min. No Optuna / GridSearchCV / transformer fine-tuning in the
core baseline. Everything seeded and reproducible.
""")

# ============================================================
# 2. IMPORTS AND CONFIGURATION
# ============================================================
md("## 2. Imports and Configuration")

code("""
import os, re, sys, time, json, hashlib, unicodedata, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

import sklearn, scipy, platform
print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("scipy:", scipy.__version__)
print("platform:", platform.platform())

try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("torch not relevant for this CPU baseline:", e)

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
        print(f"[TIMER] {self.name}: {dt:.2f}s")

RUN_T0 = time.time()
""")

# ============================================================
# 3. LOAD DATA
# ============================================================
md("""
## 3. Load Data

We discover files by walking `/kaggle/input` rather than hard-coding a path, since
API-pushed competition data can mount one or two directory levels deeper than the web UI.
""")

code("""
with Timer("data_loading"):
    DATA_FILES = {}
    for root, dirs, files in os.walk('/kaggle/input'):
        for f in files:
            if f in ('train.csv', 'test.csv', 'sample_submission.csv'):
                DATA_FILES.setdefault(f, os.path.join(root, f))
    print(DATA_FILES)
    assert len(DATA_FILES) == 3, f"Missing data files, found: {DATA_FILES}"

    train = pd.read_csv(DATA_FILES['train.csv'])
    test = pd.read_csv(DATA_FILES['test.csv'])
    sample_sub = pd.read_csv(DATA_FILES['sample_submission.csv'])

assert list(train.columns) == ['id', 'title', 'content', 'label'], train.columns.tolist()
assert list(test.columns) == ['id', 'title', 'content'], test.columns.tolist()
print("OK: schema matches expected [id,title,content(,label)]")
""")

# ============================================================
# 4. DATASET AUDIT
# ============================================================
md("## 4. Dataset Audit")

code("""
print("=== SHAPES ===")
print("train:", train.shape, " test:", test.shape)

print("\\n=== DTYPES (train) ===")
print(train.dtypes)

print("\\n=== HEAD ===")
display(train.head())
print("\\n=== TAIL ===")
display(train.tail())

print("\\n=== UNIQUE COUNTS ===")
print("unique titles (train):", train['title'].nunique(), "/", len(train))
print("unique contents (train):", train['content'].nunique(), "/", len(train))
print("unique titles (test):", test['title'].nunique(), "/", len(test))
print("unique contents (test):", test['content'].nunique(), "/", len(test))

print("\\n=== LABEL DISTRIBUTION ===")
print(train['label'].value_counts())
print(train['label'].value_counts(normalize=True))

print("\\n=== MISSING VALUES ===")
print("train:\\n", train.isna().sum())
print("test:\\n", test.isna().sum())

print("\\n=== DUPLICATE COUNTS (raw) ===")
print("duplicated id (train):", train['id'].duplicated().sum())
print("duplicated (title,content) rows (train):", train.duplicated(subset=['title','content']).sum())
""")

# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================
md("""
## 5. Exploratory Data Analysis

Every check here is tied to a modeling decision: class weighting, text-length features,
and whether validation needs to be grouped.
""")

md("### 5.1 Target distribution and majority-class Macro F1")

code("""
from sklearn.metrics import f1_score

n0 = (train['label']==0).sum()
n1 = (train['label']==1).sum()
pct0, pct1 = n0/len(train)*100, n1/len(train)*100
print(f"label=0: {n0} ({pct0:.2f}%)   label=1: {n1} ({pct1:.2f}%)")

majority_class = train['label'].mode()[0]
majority_pred = np.full(len(train), majority_class)
majority_f1 = f1_score(train['label'], majority_pred, average='macro')
print(f"Majority-class Macro F1 (predict all {majority_class}): {majority_f1:.4f}")

IMBALANCE_RATIO = max(pct0, pct1) / min(pct0, pct1)
print(f"Imbalance ratio: {IMBALANCE_RATIO:.2f}x")
if IMBALANCE_RATIO < 1.5:
    CLASS_WEIGHT_DECISION = None
    print("Decision: classes reasonably balanced -> class_weight=None as starting point.")
else:
    CLASS_WEIGHT_DECISION = 'balanced'
    print("Decision: substantial imbalance -> evaluate class_weight='balanced' (deterministic, no search).")
""")

md("### 5.2 Missing / empty values")

code("""
def is_empty(s):
    return s.isna() | (s.astype(str).str.strip() == '')

for name, df in [('train', train), ('test', test)]:
    empty_title = is_empty(df['title']).sum()
    empty_content = is_empty(df['content']).sum()
    print(f"[{name}] empty/missing title: {empty_title}   empty/missing content: {empty_content}")

# Decision: fill missing with empty string, track affected rows
train['title'] = train['title'].fillna('')
train['content'] = train['content'].fillna('')
test['title'] = test['title'].fillna('')
test['content'] = test['content'].fillna('')
print("Missing text fields filled with empty string (none expected per audit above).")
""")

md("### 5.3 Text length distributions")

code("""
with Timer("eda_lengths"):
    for df in (train, test):
        df['title_char_len'] = df['title'].str.len()
        df['content_char_len'] = df['content'].str.len()
        df['title_word_count'] = df['title'].str.split().str.len()
        df['content_word_count'] = df['content'].str.split().str.len()
        df['title_unique_word_count'] = df['title'].apply(lambda s: len(set(s.lower().split())))
        df['content_unique_word_count'] = df['content'].apply(lambda s: len(set(s.lower().split())))
        df['content_sentence_count'] = df['content'].str.count(r'[.!?]+') + 1
        df['title_digit_count'] = df['title'].str.count(r'\\d')
        df['content_digit_count'] = df['content'].str.count(r'\\d')
        df['title_punct_count'] = df['title'].str.count(r'[^\\w\\s]')
        df['content_punct_count'] = df['content'].str.count(r'[^\\w\\s]')
        df['title_content_char_ratio'] = df['title_char_len'] / df['content_char_len'].replace(0, np.nan)
        df['title_content_word_ratio'] = df['title_word_count'] / df['content_word_count'].replace(0, np.nan)

print(train[['title_char_len','content_char_len','title_word_count','content_word_count']].describe())
""")

code("""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for lbl, c in [(0, 'tab:red'), (1, 'tab:blue')]:
    sub = train[train.label == lbl]
    axes[0].hist(sub['title_word_count'].clip(upper=30), bins=30, alpha=0.5, label=f'label={lbl}', color=c, density=True)
    axes[1].hist(sub['content_word_count'].clip(upper=1000), bins=40, alpha=0.5, label=f'label={lbl}', color=c, density=True)
    axes[2].hist(sub['title_content_word_ratio'].clip(upper=0.2), bins=30, alpha=0.5, label=f'label={lbl}', color=c, density=True)
axes[0].set_title('title word count by label'); axes[0].legend()
axes[1].set_title('content word count by label'); axes[1].legend()
axes[2].set_title('title/content word ratio by label'); axes[2].legend()
plt.tight_layout()
plt.savefig('/kaggle/working/eda_lengths.png', dpi=100)
plt.close()
print("Saved eda_lengths.png")

med_title_wc = train.groupby('label')['title_word_count'].median()
med_content_wc = train.groupby('label')['content_word_count'].median()
med_ratio = train.groupby('label')['title_content_word_ratio'].median()
print("\\nMedian title word count by label:\\n", med_title_wc)
print("\\nMedian content word count by label:\\n", med_content_wc)
print("\\nMedian title/content word ratio by label:\\n", med_ratio)

print("\\nInterpretation:")
print(f"- title length differs by label: {'YES' if abs(med_title_wc.diff().iloc[-1]) > 0.5 else 'marginal'} (kept as a feature regardless, cheap).")
print(f"- content length differs by label: {'YES' if abs(med_content_wc.diff().iloc[-1]) > 5 else 'marginal'}.")
print(f"- title/content ratio differs by label: {'YES' if abs(med_ratio.diff().iloc[-1]) > 0.002 else 'marginal'}.")
""")

# ============================================================
# 6. DATA LEAKAGE / DUPLICATE ANALYSIS
# ============================================================
md("""
## 6. Data Leakage / Duplicate Analysis

If the same article body appears multiple times (re-published under different titles, or
duplicated rows), a plain random K-fold would leak that article into both train and
validation folds, inflating the score. We check this explicitly and pick the validation
scheme in Section 9 based on the result.
""")

code("""
def normalize_for_hash(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    s = re.sub(r'\\s+', ' ', s).strip()
    return s

with Timer("duplicate_analysis"):
    train['content_hash'] = train['content'].apply(lambda s: hashlib.md5(normalize_for_hash(s).encode('utf-8')).hexdigest())
    train['title_hash'] = train['title'].apply(lambda s: hashlib.md5(normalize_for_hash(s).encode('utf-8')).hexdigest())
    test['content_hash'] = test['content'].apply(lambda s: hashlib.md5(normalize_for_hash(s).encode('utf-8')).hexdigest())
    test['title_hash'] = test['title'].apply(lambda s: hashlib.md5(normalize_for_hash(s).encode('utf-8')).hexdigest())

    dup_title = train['title_hash'].duplicated(keep=False).sum()
    dup_content = train['content_hash'].duplicated(keep=False).sum()
    dup_pair = train.duplicated(subset=['title_hash','content_hash'], keep=False).sum()

    print(f"A. rows sharing a duplicated title: {dup_title}")
    print(f"B. rows sharing a duplicated content: {dup_content}")
    print(f"C. rows sharing a duplicated (title,content) pair: {dup_pair}")

    content_group_sizes = train.groupby('content_hash').size()
    multi_row_content = content_group_sizes[content_group_sizes > 1]
    print(f"D. distinct content bodies reused across >1 row: {len(multi_row_content)}")

    LEAKAGE_RISK = len(multi_row_content) > 0
    if LEAKAGE_RISK:
        n_show = 0
        for h, grp in train[train['content_hash'].isin(multi_row_content.index)].groupby('content_hash'):
            if grp['label'].nunique() > 1 or grp['title'].nunique() > 1:
                print(f"\\n-- content_hash={h[:8]} | rows={len(grp)} | unique labels={grp['label'].unique()} | unique titles={grp['title'].nunique()} --")
                display(grp[['id','title','label']].head(3))
                n_show += 1
            if n_show >= 5:
                break
        if n_show == 0:
            print("(duplicate content groups exist but all share the same title & label -> pure duplicate rows)")
    print(f"\\nLEAKAGE_RISK (duplicate content present) = {LEAKAGE_RISK}")
""")

md("### 6.1 Lightweight near-duplicate check (cheap, no all-pairs comparison)")

code("""
with Timer("near_duplicate_analysis"):
    # Cheap proxy for "same article family": hash of the first 120 normalized chars.
    # This groups near-duplicates (same lede, minor edits later in the article) without
    # any expensive pairwise similarity computation.
    train['content_prefix_hash'] = train['content'].apply(
        lambda s: hashlib.md5(normalize_for_hash(s)[:120].encode('utf-8')).hexdigest()
    )
    prefix_group_sizes = train.groupby('content_prefix_hash').size()
    near_dup_groups = (prefix_group_sizes > 1).sum()
    print(f"content-prefix groups with >1 member (near-duplicate candidates): {near_dup_groups} "
          f"(exact-content groups: {len(multi_row_content)})")
    if near_dup_groups > len(multi_row_content):
        print("-> some rows share an opening but diverge later: likely same article family, republished/edited.")
        LEAKAGE_RISK = True
""")

md("### 6.2 Vocabulary / publisher-boilerplate audit")

code("""
BOILERPLATE_PATTERNS = ['kumparan com', 'detik com', 'kompas com', 'liputan6 com', 'tempo co',
                         'cnn indonesia', 'cnbc indonesia', 'tribunnews com', 'merdeka com',
                         'okezone com', 'sindonews com', 'antaranews com', 'republika co id',
                         'suara com', 'viva co id']
sample_text = ' '.join(train['content'].str.lower().sample(min(3000, len(train)), random_state=SEED))
for pat in BOILERPLATE_PATTERNS:
    cnt = sample_text.count(pat)
    if cnt > 0:
        print(f"boilerplate pattern '{pat}': ~{cnt} occurrences in 3000-row sample")

has_url = train['content'].str.contains(r'https?://|www\\.', regex=True).sum()
has_html = train['content'].str.contains(r'<[a-zA-Z]+[^>]*>', regex=True).sum()
has_digit_title = train['title'].str.contains(r'\\d').sum()
print(f"\\nrows with URL-like text in content: {has_url}")
print(f"rows with HTML-like tags in content: {has_html}")
print(f"rows with digits in title: {has_digit_title} ({has_digit_title/len(train)*100:.1f}%)")
print("\\nDecision: publisher-suffix boilerplate (e.g. 'kumparan com') is noise for the")
print("word-level signal and will be stripped in the *normalized* text view only; the raw")
print("view is kept untouched for character-level features and to avoid destroying")
print("legitimate numbers/entities/punctuation.")
""")

# ============================================================
# 7. TEXT PREPROCESSING
# ============================================================
md("""
## 7. Text Preprocessing

We keep **multiple** representations instead of one destructive pipeline:
`title_raw`, `content_raw` (untouched) and `title_norm`, `content_norm` (unicode-normalized,
whitespace-collapsed, URL/HTML/publisher-boilerplate stripped, lower-cased). No stemming,
no lemmatization, no global number/entity/punctuation removal — those can carry the exact
factual-consistency signal we need (numbers, negation, named entities).
""")

code("""
URL_RE = re.compile(r'https?://\\S+|www\\.\\S+')
HTML_RE = re.compile(r'<[^>]+>')
WS_RE = re.compile(r'\\s+')
BOILERPLATE_RE = re.compile(
    r'\\b(kumparan|detik|kompas|liputan6|tempo|cnn indonesia|cnbc indonesia|tribunnews|'
    r'merdeka|okezone|sindonews|antaranews|republika|suara|viva)\\b[\\s.]*(com|co id)?',
    re.IGNORECASE)

def normalize_text(s):
    if not isinstance(s, str):
        return ''
    s = unicodedata.normalize('NFKC', s)
    s = HTML_RE.sub(' ', s)
    s = URL_RE.sub(' ', s)
    s = BOILERPLATE_RE.sub(' ', s)
    s = WS_RE.sub(' ', s).strip()
    return s

def normalize_lower(s):
    return normalize_text(s).lower()

with Timer("preprocessing"):
    for df in (train, test):
        df['title_raw'] = df['title'].astype(str)
        df['content_raw'] = df['content'].astype(str)
        df['title_norm'] = df['title_raw'].apply(normalize_lower)
        df['content_norm'] = df['content_raw'].apply(normalize_lower)

print(train[['title_raw','title_norm']].head(3))
""")

md("### 7.1 Number extraction (kept separate, not blindly tokenized away)")

code("""
NUM_RE = re.compile(r'\\d+[.,]?\\d*\\s*%?')

def extract_numbers(s):
    return NUM_RE.findall(s)

with Timer("number_extraction"):
    for df in (train, test):
        df['title_numbers'] = df['title_raw'].apply(extract_numbers)
        df['content_numbers'] = df['content_raw'].apply(extract_numbers)

print("Example numbers extracted from a title:", train['title_numbers'].iloc[1])
print("NOTE: source text has thousand separators already stripped (e.g. '1 665' instead of")
print("'1.665'), so multi-digit grouped numbers may appear as separate tokens. Number")
print("features below are therefore approximate token-level signals, not exact parses.")
""")

# ============================================================
# 8. RELATIONSHIP FEATURE ENGINEERING
# ============================================================
md("""
## 8. Relationship Feature Engineering

Handcrafted pair-level features capturing token overlap, numeric agreement and negation —
the same feature families used in Fake News Challenge-style stance detection. These are
purely a function of `(title, content)` text and do not use the label, so they can be
computed once globally (no leakage) and reused as **B2/B3 inputs**; TF-IDF-derived cosine
similarities are instead fit **inside each CV fold** in Section 9 to avoid vocabulary
leakage from the validation fold.

Named-entity consistency (Section 4.7 of the spec) is **skipped**: no Indonesian NER model
ships with the base Kaggle image and downloading/loading one would jeopardize the runtime
budget for a baseline; this is documented rather than silently omitted.
""")

code("""
NEGATION_WORDS = ['tidak','bukan','belum','tanpa','gagal','ditolak','membantah','bantah',
                  'sangkal','menyangkal']

def word_set(s):
    return set(re.findall(r'\\w+', s.lower()))

def word_list(s):
    return re.findall(r'\\w+', s.lower())

def safe_div(a, b):
    return a / b if b else 0.0

def bigrams(tokens):
    return set(zip(tokens, tokens[1:])) if len(tokens) > 1 else set()

def negation_count(s):
    toks = word_list(s)
    return sum(1 for t in toks if t in NEGATION_WORDS)

def build_pair_features(df):
    feats = []
    for title, content in zip(df['title_norm'], df['content_norm']):
        tw_list, cw_list = word_list(title), word_list(content)
        tw, cw = set(tw_list), set(cw_list)
        inter = tw & cw
        union = tw | cw
        tbg, cbg = bigrams(tw_list), bigrams(cw_list)
        bigram_inter = tbg & cbg
        f = {
            'n_shared_words': len(inter),
            'title_coverage': safe_div(len(inter), len(tw)),
            'content_coverage': safe_div(len(inter), len(cw)),
            'jaccard_words': safe_div(len(inter), len(union)),
            'bigram_jaccard': safe_div(len(bigram_inter), len(tbg | cbg)) if (tbg or cbg) else 0.0,
            'title_word_count_f': len(tw_list),
            'content_word_count_f': len(cw_list),
            'title_char_len_f': len(title),
            'content_char_len_f': len(content),
            'char_ratio': safe_div(len(title), len(content)),
            'word_ratio': safe_div(len(tw_list), len(cw_list)),
            'log_content_len': np.log1p(len(content)),
            'title_negation_count': negation_count(title),
            'content_negation_count': negation_count(content),
        }
        f['negation_mismatch'] = float((f['title_negation_count'] > 0) != (f['content_negation_count'] > 0))
        feats.append(f)
    return pd.DataFrame(feats, index=df.index)

with Timer("pair_features"):
    train_pair_feats = build_pair_features(train)
    test_pair_feats = build_pair_features(test)

print(train_pair_feats.shape, test_pair_feats.shape)
train_pair_feats.describe().T
""")

md("### 8.1 Number consistency features")

code("""
def build_number_features(df):
    feats = []
    for tnums, cnums in zip(df['title_numbers'], df['content_numbers']):
        tset, cset = set(tnums), set(cnums)
        inter = tset & cset
        f = {
            'n_numbers_title': len(tnums),
            'n_numbers_content': len(cnums),
            'n_numbers_shared': len(inter),
            'number_coverage': safe_div(len(inter), len(tset)),
            'has_title_numbers': float(len(tset) > 0),
            'number_mismatch': float(len(tset) > 0 and len(inter) == 0),
        }
        feats.append(f)
    return pd.DataFrame(feats, index=df.index)

with Timer("number_features"):
    train_num_feats = build_number_features(train)
    test_num_feats = build_number_features(test)

print("Number-mismatch rate by label:")
print(pd.concat([train_num_feats['number_mismatch'], train['label']], axis=1).groupby('label').mean())
""")

# ============================================================
# 9. VALIDATION STRATEGY
# ============================================================
md("""
## 9. Validation Strategy

Decision rule (per Section 6 findings): if any content body is duplicated across rows,
plain `StratifiedKFold` risks leaking that article across train/val, so we switch to
`StratifiedGroupKFold` grouped on `content_hash`. Otherwise `StratifiedKFold` is used.
Both use 5 splits, `shuffle=True`, `random_state=42`. No vectorizer is ever fit on data
outside its own training fold.
""")

code("""
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

N_SPLITS = 5
X_dummy = np.zeros(len(train))
y = train['label'].values
groups = train['content_hash'].values

if LEAKAGE_RISK:
    VALIDATION_METHOD = 'StratifiedGroupKFold'
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_iter = list(splitter.split(X_dummy, y, groups))
else:
    VALIDATION_METHOD = 'StratifiedKFold'
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_iter = list(splitter.split(X_dummy, y))

print(f"Validation method selected: {VALIDATION_METHOD} (n_splits={N_SPLITS})")
for i, (tr_idx, va_idx) in enumerate(fold_iter):
    print(f"fold {i}: train={len(tr_idx)} val={len(va_idx)} val_label_mean={y[va_idx].mean():.3f}")
    if LEAKAGE_RISK:
        overlap = set(groups[tr_idx]) & set(groups[va_idx])
        assert len(overlap) == 0, f"content_hash leakage between train/val in fold {i}!"
print("No content_hash overlap between train/val folds confirmed." if LEAKAGE_RISK else "Folds stratified on label only (no duplicate-content leakage risk detected).")
""")

# ============================================================
# 10-13. BASELINES B0-B3
# ============================================================
md("""
## 10-13. Baselines B0-B3

All four baselines are evaluated with the same fold assignment (`fold_iter`) from Section 9,
so their OOF Macro F1 scores are directly comparable. Every TF-IDF vectorizer is fit
**inside** the training portion of each fold only.
""")

code("""
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import scipy.sparse as sp

oof_preds = {}
oof_scores = {}
fold_scores = {}
model_runtime = {}

def report_model(name, y_true, y_pred, fold_f1s, runtime):
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    oof_scores[name] = macro_f1
    fold_scores[name] = fold_f1s
    model_runtime[name] = runtime
    print(f"[{name}] pooled OOF Macro F1 = {macro_f1:.4f} | mean fold F1 = {np.mean(fold_f1s):.4f} "
          f"+/- {np.std(fold_f1s):.4f} | runtime = {runtime:.1f}s")
""")

md("### B0 — Majority-class sanity check")

code("""
with Timer("B0_majority") as t0_timer:
    t0 = time.time()
    b0_pred = np.full(len(train), train['label'].mode()[0])
    fold_f1s = []
    for tr_idx, va_idx in fold_iter:
        f1 = f1_score(y[va_idx], b0_pred[va_idx], average='macro')
        fold_f1s.append(f1)
    runtime = time.time() - t0
oof_preds['B0_majority'] = b0_pred
report_model('B0_majority', y, b0_pred, fold_f1s, runtime)
""")

md("""
### B1 — Word TF-IDF (title + content, separate vectorizers) + LinearSVC

Fixed config (no search): `ngram_range=(1,2)`, `min_df=2`, `sublinear_tf=True`,
`strip_accents='unicode'`, `norm='l2'`. Title and content are vectorized independently and
concatenated (`hstack`) — the model is never shown a merged title+body string, so it can
learn title-specific vs. content-specific signal separately.
""")

code("""
def make_word_tfidf(max_features):
    return TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True,
                            strip_accents='unicode', norm='l2', max_features=max_features)

with Timer("B1_word_tfidf_svm"):
    t0 = time.time()
    b1_oof = np.zeros(len(train), dtype=int)
    fold_f1s = []
    for fold_i, (tr_idx, va_idx) in enumerate(fold_iter):
        title_vec = make_word_tfidf(20000)
        content_vec = make_word_tfidf(40000)
        Xtr_title = title_vec.fit_transform(train['title_norm'].iloc[tr_idx])
        Xva_title = title_vec.transform(train['title_norm'].iloc[va_idx])
        Xtr_content = content_vec.fit_transform(train['content_norm'].iloc[tr_idx])
        Xva_content = content_vec.transform(train['content_norm'].iloc[va_idx])

        Xtr = sp.hstack([Xtr_title, Xtr_content]).tocsr()
        Xva = sp.hstack([Xva_title, Xva_content]).tocsr()

        clf = LinearSVC(class_weight=CLASS_WEIGHT_DECISION, random_state=SEED)
        clf.fit(Xtr, y[tr_idx])
        pred = clf.predict(Xva)
        b1_oof[va_idx] = pred
        f1 = f1_score(y[va_idx], pred, average='macro')
        fold_f1s.append(f1)
        print(f"  fold {fold_i}: Macro F1 = {f1:.4f}")
    runtime = time.time() - t0
oof_preds['B1_word_tfidf_svm'] = b1_oof
report_model('B1_word_tfidf_svm', y, b1_oof, fold_f1s, runtime)
""")

md("""
### B2 — Word + Char TF-IDF + Handcrafted Pair Features + LinearSVC (MAIN BASELINE)

Adds character n-gram TF-IDF (`analyzer='char'`, `ngram_range=(3,5)`) for both title and
content (robust to typos/morphology) plus the handcrafted relationship features from
Section 8, converted to sparse and `hstack`-ed alongside the TF-IDF blocks. Feature caps
are fixed (not tuned) purely to keep runtime bounded.
""")

code("""
def make_char_tfidf(max_features):
    return TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=3, sublinear_tf=True,
                            max_features=max_features)

pair_feature_cols = list(train_pair_feats.columns) + list(train_num_feats.columns)
all_pair_feats_train = pd.concat([train_pair_feats, train_num_feats], axis=1).values
all_pair_feats_test = pd.concat([test_pair_feats, test_num_feats], axis=1).values

with Timer("B2_word_char_tfidf_pairfeats_svm"):
    t0 = time.time()
    b2_oof = np.zeros(len(train), dtype=int)
    b2_oof_scores_arr = np.zeros(len(train), dtype=float)
    fold_f1s = []
    for fold_i, (tr_idx, va_idx) in enumerate(fold_iter):
        tw_vec = make_word_tfidf(20000)
        cw_vec = make_word_tfidf(40000)
        tc_vec = make_char_tfidf(20000)
        cc_vec = make_char_tfidf(30000)

        Xtr_tw = tw_vec.fit_transform(train['title_norm'].iloc[tr_idx])
        Xva_tw = tw_vec.transform(train['title_norm'].iloc[va_idx])
        Xtr_cw = cw_vec.fit_transform(train['content_norm'].iloc[tr_idx])
        Xva_cw = cw_vec.transform(train['content_norm'].iloc[va_idx])
        Xtr_tc = tc_vec.fit_transform(train['title_norm'].iloc[tr_idx])
        Xva_tc = tc_vec.transform(train['title_norm'].iloc[va_idx])
        Xtr_cc = cc_vec.fit_transform(train['content_norm'].iloc[tr_idx])
        Xva_cc = cc_vec.transform(train['content_norm'].iloc[va_idx])

        Xtr_pf = sp.csr_matrix(all_pair_feats_train[tr_idx])
        Xva_pf = sp.csr_matrix(all_pair_feats_train[va_idx])

        Xtr = sp.hstack([Xtr_tw, Xtr_cw, Xtr_tc, Xtr_cc, Xtr_pf]).tocsr()
        Xva = sp.hstack([Xva_tw, Xva_cw, Xva_tc, Xva_cc, Xva_pf]).tocsr()

        clf = LinearSVC(class_weight=CLASS_WEIGHT_DECISION, random_state=SEED)
        clf.fit(Xtr, y[tr_idx])
        pred = clf.predict(Xva)
        score = clf.decision_function(Xva)
        b2_oof[va_idx] = pred
        b2_oof_scores_arr[va_idx] = score
        f1 = f1_score(y[va_idx], pred, average='macro')
        fold_f1s.append(f1)
        print(f"  fold {fold_i}: Macro F1 = {f1:.4f}")
    runtime = time.time() - t0
oof_preds['B2_word_char_tfidf_pairfeats_svm'] = b2_oof
report_model('B2_word_char_tfidf_pairfeats_svm', y, b2_oof, fold_f1s, runtime)
B2_OOF_F1 = oof_scores['B2_word_char_tfidf_pairfeats_svm']
""")

md("""
### B3 — Relationship-features-only model (interpretability baseline)

A compact dense table: handcrafted overlap/number/negation features from Section 8, plus a
fold-safe TF-IDF cosine similarity (word- and char-level) between title and content, fit
fresh inside every fold on a *shared* title+content vocabulary (so cosine is meaningful).
This model is not expected to beat B2 — its purpose is to show how much of the task a pure
relationship signal (no raw vocabulary) can solve.
""")

code("""
from sklearn.metrics.pairwise import cosine_similarity

def fold_safe_cosine(train_titles, train_contents, va_titles, va_contents, analyzer, ngram_range, max_features, min_df):
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df,
                           sublinear_tf=True, max_features=max_features)
    fit_corpus = pd.concat([train_titles, train_contents], axis=0)
    vec.fit(fit_corpus)
    def cos_for(titles, contents):
        Tt = vec.transform(titles)
        Tc = vec.transform(contents)
        # row-wise cosine between corresponding title/content vectors
        num = np.asarray(Tt.multiply(Tc).sum(axis=1)).ravel()
        tnorm = np.sqrt(np.asarray(Tt.multiply(Tt).sum(axis=1)).ravel())
        cnorm = np.sqrt(np.asarray(Tc.multiply(Tc).sum(axis=1)).ravel())
        denom = tnorm * cnorm
        denom[denom == 0] = 1.0
        return num / denom
    return cos_for(train_titles, train_contents), cos_for(va_titles, va_contents)

with Timer("B3_relationship_only"):
    t0 = time.time()
    b3_oof = np.zeros(len(train), dtype=int)
    fold_f1s = []
    for fold_i, (tr_idx, va_idx) in enumerate(fold_iter):
        word_cos_tr, word_cos_va = fold_safe_cosine(
            train['title_norm'].iloc[tr_idx], train['content_norm'].iloc[tr_idx],
            train['title_norm'].iloc[va_idx], train['content_norm'].iloc[va_idx],
            analyzer='word', ngram_range=(1,2), max_features=20000, min_df=2)
        char_cos_tr, char_cos_va = fold_safe_cosine(
            train['title_norm'].iloc[tr_idx], train['content_norm'].iloc[tr_idx],
            train['title_norm'].iloc[va_idx], train['content_norm'].iloc[va_idx],
            analyzer='char', ngram_range=(3,5), max_features=20000, min_df=3)

        Xtr = np.column_stack([all_pair_feats_train[tr_idx], word_cos_tr, char_cos_tr])
        Xva = np.column_stack([all_pair_feats_train[va_idx], word_cos_va, char_cos_va])

        clf = LogisticRegression(class_weight=CLASS_WEIGHT_DECISION, max_iter=1000, random_state=SEED)
        clf.fit(Xtr, y[tr_idx])
        pred = clf.predict(Xva)
        b3_oof[va_idx] = pred
        f1 = f1_score(y[va_idx], pred, average='macro')
        fold_f1s.append(f1)
        print(f"  fold {fold_i}: Macro F1 = {f1:.4f}")
    runtime = time.time() - t0
oof_preds['B3_relationship_only'] = b3_oof
report_model('B3_relationship_only', y, b3_oof, fold_f1s, runtime)
""")

# ============================================================
# 14. OOF EVALUATION AND MODEL COMPARISON
# ============================================================
md("## 14. OOF Evaluation and Model Comparison")

code("""
leaderboard = pd.DataFrame([
    {'Model': name, 'OOF Macro F1': oof_scores[name], 'Mean Fold F1': np.mean(fold_scores[name]),
     'Std': np.std(fold_scores[name]), 'Runtime (s)': model_runtime[name]}
    for name in oof_scores
]).sort_values('OOF Macro F1', ascending=False).reset_index(drop=True)
print(leaderboard.to_string(index=False))

BEST_MODEL = leaderboard.iloc[0]['Model']
print(f"\\nSelected model (highest OOF Macro F1, runtime as tiebreaker): {BEST_MODEL}")

print("\\nConfusion matrix (OOF, best model):")
print(confusion_matrix(y, oof_preds[BEST_MODEL]))
print(classification_report(y, oof_preds[BEST_MODEL], digits=4))
""")

# ============================================================
# 15. ERROR ANALYSIS
# ============================================================
md("""
## 15. Error Analysis

Using OOF predictions from the best baseline (expected: B2, which also has a
`decision_function` score usable as a confidence proxy).
""")

code("""
with Timer("error_analysis"):
    if BEST_MODEL == 'B2_word_char_tfidf_pairfeats_svm':
        conf_scores = b2_oof_scores_arr
    else:
        conf_scores = np.zeros(len(train))  # fallback: no calibrated score available

    err_df = train[['id','title','content','label']].copy()
    err_df['pred'] = oof_preds[BEST_MODEL]
    err_df['score'] = conf_scores
    err_df['content_excerpt'] = err_df['content'].str.slice(0, 220)
    err_df['is_error'] = err_df['label'] != err_df['pred']

    fp = err_df[(err_df.label == 0) & (err_df.pred == 1)].copy()
    fn = err_df[(err_df.label == 1) & (err_df.pred == 0)].copy()
    print(f"False positives (true=0, pred=1): {len(fp)}")
    print(f"False negatives (true=1, pred=0): {len(fn)}")

    highest_conf_mistakes = err_df[err_df.is_error].reindex(err_df[err_df.is_error].score.abs().sort_values(ascending=False).index).head(10)
    lowest_conf_preds = err_df.reindex(err_df.score.abs().sort_values(ascending=True).index).head(10)

    pd.set_option('display.max_colwidth', 150)
    print("\\n=== Highest-confidence mistakes ===")
    display(highest_conf_mistakes[['id','title','content_excerpt','label','pred','score']])
    print("\\n=== Lowest-confidence predictions (near decision boundary) ===")
    display(lowest_conf_preds[['id','title','content_excerpt','label','pred','score']])

    print(f"\\nTotal displayed examples: {len(highest_conf_mistakes) + len(lowest_conf_preds)} "
          f"(target: >=20)")
""")

code("""
# Manual-style categorization heuristics (rule-of-thumb, not ground truth) to estimate
# the *proportion* of error types worth investing further modeling effort into.
def categorize_error(row):
    t, c = row['title'].lower(), row['content'].lower()
    tnum = set(NUM_RE.findall(row['title']))
    cnum = set(NUM_RE.findall(row['content']))
    if tnum and not (tnum & cnum):
        return 'number_mismatch'
    if any(w in t for w in NEGATION_WORDS) != any(w in c for w in NEGATION_WORDS):
        return 'negation_contradiction'
    tw, cw = word_set(t), word_set(c)
    jac = safe_div(len(tw & cw), len(tw | cw))
    if jac < 0.05:
        return 'same_topic_different_event_or_lexical_mismatch'
    if jac > 0.4:
        return 'semantic_paraphrase_or_ambiguous'
    return 'ambiguous_hard_case'

error_rows = err_df[err_df.is_error].copy()
error_rows['category'] = error_rows.apply(categorize_error, axis=1)
cat_counts = error_rows['category'].value_counts(normalize=True) * 100
print("Estimated error category breakdown (heuristic, on OOF errors of the best model):")
print(cat_counts.round(1))
""")

# ============================================================
# 16. OPTIONAL SEMANTIC MODEL
# ============================================================
md("""
## 16. Optional Fast Semantic Baseline (conditional)

Only runs if **B2 OOF Macro F1 < 0.90**. Uses a small frozen multilingual sentence-embedding
model (no fine-tuning) to test whether semantic similarity adds value over lexical TF-IDF.
Article bodies are truncated to their first few sentences before encoding (never the full
body) to keep inference fast.
""")

code("""
RUN_SEMANTIC = B2_OOF_F1 < 0.90
print(f"B2 OOF Macro F1 = {B2_OOF_F1:.4f}  ->  RUN_SEMANTIC = {RUN_SEMANTIC}")

SEMANTIC_OOF_F1 = None
if RUN_SEMANTIC:
    with Timer("optional_semantic_model"):
        t0 = time.time()
        try:
            import importlib
            if importlib.util.find_spec('sentence_transformers') is None:
                os.system('pip install -q sentence-transformers')
            from sentence_transformers import SentenceTransformer

            def first_n_sentences(s, n=5):
                parts = re.split(r'(?<=[.!?])\\s+', s)
                return ' '.join(parts[:n])

            content_short = train['content_norm'].apply(lambda s: first_n_sentences(s, 5))
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            title_emb = model.encode(train['title_norm'].tolist(), batch_size=128, show_progress_bar=False)
            content_emb = model.encode(content_short.tolist(), batch_size=128, show_progress_bar=False)

            cos = np.sum(title_emb * content_emb, axis=1) / (
                np.linalg.norm(title_emb, axis=1) * np.linalg.norm(content_emb, axis=1) + 1e-8)
            abs_diff = np.abs(title_emb - content_emb)
            prod = title_emb * content_emb

            sem_feats = np.column_stack([cos, abs_diff, prod, all_pair_feats_train])

            sem_oof = np.zeros(len(train), dtype=int)
            fold_f1s = []
            for fold_i, (tr_idx, va_idx) in enumerate(fold_iter):
                clf = LogisticRegression(max_iter=1000, class_weight=CLASS_WEIGHT_DECISION, random_state=SEED)
                clf.fit(sem_feats[tr_idx], y[tr_idx])
                pred = clf.predict(sem_feats[va_idx])
                sem_oof[va_idx] = pred
                fold_f1s.append(f1_score(y[va_idx], pred, average='macro'))
            runtime = time.time() - t0
            oof_preds['B4_semantic_frozen'] = sem_oof
            report_model('B4_semantic_frozen', y, sem_oof, fold_f1s, runtime)
            SEMANTIC_OOF_F1 = oof_scores['B4_semantic_frozen']

            delta = SEMANTIC_OOF_F1 - B2_OOF_F1
            print(f"\\nSemantic vs B2 delta: {delta:+.4f}")
            if delta < 0.01:
                print("Improvement is small -> REJECT semantic model, keep B2 as baseline.")
            else:
                print("Meaningful improvement -> semantic features are worth keeping.")
        except Exception as e:
            print(f"Semantic baseline skipped due to environment/runtime issue: {e}")
else:
    print("B2 already >= 0.90 Macro F1 -> skipping optional semantic model per spec.")
""")

# ============================================================
# 17. FINAL MODEL TRAINING
# ============================================================
md("""
## 17. Final Model Training

Refit the selected model's full pipeline (preprocessing already global/leakage-free;
TF-IDF vectorizers + classifier) on **all** training data, then transform test.
""")

code("""
with Timer("final_training"):
    t0 = time.time()
    FINAL_MODEL_NAME = BEST_MODEL if SEMANTIC_OOF_F1 is None or SEMANTIC_OOF_F1 - B2_OOF_F1 < 0.01 else 'B4_semantic_frozen'

    _HANDLED_MODELS = {'B1_word_tfidf_svm', 'B2_word_char_tfidf_pairfeats_svm',
                        'B3_relationship_only', 'B4_semantic_frozen'}
    if FINAL_MODEL_NAME not in _HANDLED_MODELS:
        print(f"[WARNING] Best model '{FINAL_MODEL_NAME}' has no dedicated final-training "
              f"path (expected only on tiny/degenerate data, e.g. B0 winning by chance). "
              f"Falling back to B2, the designated main baseline.")
        FINAL_MODEL_NAME = 'B2_word_char_tfidf_pairfeats_svm'

    print(f"Final model for inference: {FINAL_MODEL_NAME}")

    if FINAL_MODEL_NAME == 'B2_word_char_tfidf_pairfeats_svm':
        tw_vec = make_word_tfidf(20000).fit(train['title_norm'])
        cw_vec = make_word_tfidf(40000).fit(train['content_norm'])
        tc_vec = make_char_tfidf(20000).fit(train['title_norm'])
        cc_vec = make_char_tfidf(30000).fit(train['content_norm'])

        Xtr_full = sp.hstack([
            tw_vec.transform(train['title_norm']),
            cw_vec.transform(train['content_norm']),
            tc_vec.transform(train['title_norm']),
            cc_vec.transform(train['content_norm']),
            sp.csr_matrix(all_pair_feats_train),
        ]).tocsr()

        final_clf = LinearSVC(class_weight=CLASS_WEIGHT_DECISION, random_state=SEED)
        final_clf.fit(Xtr_full, y)
    elif FINAL_MODEL_NAME == 'B1_word_tfidf_svm':
        title_vec = make_word_tfidf(20000).fit(train['title_norm'])
        content_vec = make_word_tfidf(40000).fit(train['content_norm'])
        Xtr_full = sp.hstack([title_vec.transform(train['title_norm']), content_vec.transform(train['content_norm'])]).tocsr()
        final_clf = LinearSVC(class_weight=CLASS_WEIGHT_DECISION, random_state=SEED)
        final_clf.fit(Xtr_full, y)
    elif FINAL_MODEL_NAME == 'B3_relationship_only':
        word_cos_tr, _ = fold_safe_cosine(train['title_norm'], train['content_norm'],
                                           train['title_norm'].iloc[:1], train['content_norm'].iloc[:1],
                                           'word', (1,2), 20000, 2)
        char_cos_tr, _ = fold_safe_cosine(train['title_norm'], train['content_norm'],
                                           train['title_norm'].iloc[:1], train['content_norm'].iloc[:1],
                                           'char', (3,5), 20000, 3)
        Xtr_full = np.column_stack([all_pair_feats_train, word_cos_tr, char_cos_tr])
        final_clf = LogisticRegression(class_weight=CLASS_WEIGHT_DECISION, max_iter=1000, random_state=SEED)
        final_clf.fit(Xtr_full, y)
    else:  # FINAL_MODEL_NAME == 'B4_semantic_frozen' (only reachable if RUN_SEMANTIC was True)
        content_short_tr = train['content_norm'].apply(lambda s: first_n_sentences(s, 5))
        title_emb_tr = model.encode(train['title_norm'].tolist(), batch_size=128, show_progress_bar=False)
        content_emb_tr = model.encode(content_short_tr.tolist(), batch_size=128, show_progress_bar=False)
        cos_tr = np.sum(title_emb_tr * content_emb_tr, axis=1) / (
            np.linalg.norm(title_emb_tr, axis=1) * np.linalg.norm(content_emb_tr, axis=1) + 1e-8)
        Xtr_full = np.column_stack([cos_tr, np.abs(title_emb_tr - content_emb_tr),
                                     title_emb_tr * content_emb_tr, all_pair_feats_train])
        final_clf = LogisticRegression(max_iter=1000, class_weight=CLASS_WEIGHT_DECISION, random_state=SEED)
        final_clf.fit(Xtr_full, y)
        FINAL_SEMANTIC_MODEL = model
    runtime = time.time() - t0
print(f"Final model trained in {runtime:.1f}s")
""")

# ============================================================
# 18. TEST INFERENCE
# ============================================================
md("## 18. Test Inference")

code("""
with Timer("test_inference"):
    t0 = time.time()
    if FINAL_MODEL_NAME == 'B2_word_char_tfidf_pairfeats_svm':
        Xte_full = sp.hstack([
            tw_vec.transform(test['title_norm']),
            cw_vec.transform(test['content_norm']),
            tc_vec.transform(test['title_norm']),
            cc_vec.transform(test['content_norm']),
            sp.csr_matrix(all_pair_feats_test),
        ]).tocsr()
    elif FINAL_MODEL_NAME == 'B1_word_tfidf_svm':
        Xte_full = sp.hstack([title_vec.transform(test['title_norm']), content_vec.transform(test['content_norm'])]).tocsr()
    elif FINAL_MODEL_NAME == 'B3_relationship_only':
        word_cos_te, _ = fold_safe_cosine(train['title_norm'], train['content_norm'],
                                           test['title_norm'], test['content_norm'],
                                           'word', (1,2), 20000, 2)
        char_cos_te, _ = fold_safe_cosine(train['title_norm'], train['content_norm'],
                                           test['title_norm'], test['content_norm'],
                                           'char', (3,5), 20000, 3)
        Xte_full = np.column_stack([all_pair_feats_test, word_cos_te, char_cos_te])
    elif FINAL_MODEL_NAME == 'B4_semantic_frozen':
        content_short_te = test['content_norm'].apply(lambda s: first_n_sentences(s, 5))
        title_emb_te = FINAL_SEMANTIC_MODEL.encode(test['title_norm'].tolist(), batch_size=128, show_progress_bar=False)
        content_emb_te = FINAL_SEMANTIC_MODEL.encode(content_short_te.tolist(), batch_size=128, show_progress_bar=False)
        cos_te = np.sum(title_emb_te * content_emb_te, axis=1) / (
            np.linalg.norm(title_emb_te, axis=1) * np.linalg.norm(content_emb_te, axis=1) + 1e-8)
        Xte_full = np.column_stack([cos_te, np.abs(title_emb_te - content_emb_te),
                                     title_emb_te * content_emb_te, all_pair_feats_test])

    test_pred = final_clf.predict(Xte_full)
    inference_runtime = time.time() - t0
print(f"Inference done in {inference_runtime:.1f}s")
print(pd.Series(test_pred).value_counts(normalize=True))
""")

# ============================================================
# 19. SUBMISSION CREATION
# ============================================================
md("## 19. Submission Creation")

code("""
submission = pd.DataFrame({'id': test['id'], 'label': test_pred.astype(int)})
submission = submission.set_index('id').loc[test['id']].reset_index()

assert submission.shape[0] == len(test)
assert set(submission['id']) == set(sample_sub['id'])
assert set(submission['label'].unique()).issubset({0, 1})
assert submission['label'].isna().sum() == 0
assert list(submission.columns) == ['id', 'label']

submission.to_csv('/kaggle/working/submission.csv', index=False)
print(submission.head())
print(submission.shape)
print(submission['label'].value_counts(normalize=True))
print("\\nsubmission.csv written to /kaggle/working/ (NOT auto-submitted to the leaderboard).")
""")

# ============================================================
# 20. FINAL RUNTIME AND RESULTS SUMMARY
# ============================================================
md("## 20. Final Runtime and Results Summary")

code("""
TOTAL_RUNTIME = time.time() - RUN_T0

print("=" * 48)
print("FINAL BASELINE SUMMARY")
print(f"Dataset:\\n  Train = {len(train)}\\n  Test = {len(test)}")
print(f"Class distribution:\\n  0 = {pct0:.1f}%\\n  1 = {pct1:.1f}%")
print(f"Validation:\\n  Method = {VALIDATION_METHOD}\\n  Folds = {N_SPLITS}")
print("Results:")
for name in oof_scores:
    print(f"  {name}: Macro F1 = {oof_scores[name]:.4f}")
if SEMANTIC_OOF_F1 is not None:
    print(f"  Optional semantic model: Macro F1 = {SEMANTIC_OOF_F1:.4f}")
else:
    print("  Optional semantic model: not run (B2 already >= 0.90 or skipped)")
print(f"Selected model:\\n  {FINAL_MODEL_NAME}")
print(f"OOF Macro F1:\\n  {oof_scores.get(BEST_MODEL, float('nan')):.4f}")
print(f"Training runtime:\\n  {model_runtime.get(BEST_MODEL, float('nan')):.1f} seconds (CV) ")
print(f"Inference runtime:\\n  {inference_runtime:.1f} seconds")
print(f"Total runtime:\\n  {TOTAL_RUNTIME:.1f} seconds ({TOTAL_RUNTIME/60:.1f} min)")
print("Submission:\\n  submission.csv")
print("=" * 48)

print("\\nPer-component timings:")
for k, v in sorted(TIMINGS.items(), key=lambda kv: -kv[1]):
    print(f"  {k}: {v:.1f}s")

if TOTAL_RUNTIME > 20*60:
    print("\\n[WARNING] Total runtime exceeded the 20-minute hard budget.")
elif TOTAL_RUNTIME > 10*60:
    print("\\n[NOTE] Total runtime exceeded the 10-minute target but is within the 20-minute hard budget.")
else:
    print("\\nRuntime within the 10-minute target.")
""")

nb['cells'] = cells
nb['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb['metadata']['language_info'] = {'name': 'python', 'version': '3.11'}

with open('ifest2026_dac_fast_baseline.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook written.")
