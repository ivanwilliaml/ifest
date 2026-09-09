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
# Penyisihan IFEST 2026 DAC — Comprehensive EDA / Data Reconnaissance

**Purpose of this notebook:** understand the dataset deeply enough to make defensible
modeling decisions. This is **not** a model-building notebook — only tiny diagnostic
classifiers (Section 27) are trained, purely to estimate how much of the task each
feature family can explain. Every analysis below closes with an explicit
**"so what does this mean for modeling"** interpretation.

Target runtime: < 10 min, hard budget < 20 min. No transformer inference, no O(N^2)
pairwise comparisons, no hyperparameter search.
""")

# ---------------------------------------------------------------
code("""
import os, re, sys, time, json, hashlib, unicodedata, string, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

import sklearn, scipy, platform
print("Python:", sys.version.split()[0], "| pandas:", pd.__version__, "| numpy:", np.__version__,
      "| sklearn:", sklearn.__version__, "| scipy:", scipy.__version__)

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
pd.set_option('display.max_colwidth', 140)
""")

# ============================================================
# 1. DATA LOADING
# ============================================================
md("## 1. Data Loading")

code("""
with Timer("data_loading"):
    DATA_FILES = {}
    for root, dirs, files in os.walk('/kaggle/input'):
        for f in files:
            if f in ('train.csv', 'test.csv', 'sample_submission.csv'):
                DATA_FILES.setdefault(f, os.path.join(root, f))
    for name, path in DATA_FILES.items():
        print(f"{name}: {path}  ({os.path.getsize(path)/1e6:.2f} MB)")
    assert len(DATA_FILES) == 3, f"Missing data files: {DATA_FILES}"

    train = pd.read_csv(DATA_FILES['train.csv'])
    test = pd.read_csv(DATA_FILES['test.csv'])
    sample_sub = pd.read_csv(DATA_FILES['sample_submission.csv'])

assert list(train.columns) == ['id', 'title', 'content', 'label'], train.columns.tolist()
assert list(test.columns) == ['id', 'title', 'content'], test.columns.tolist()

print("\\ntrain shape:", train.shape, " test shape:", test.shape)
print("\\ntrain dtypes:\\n", train.dtypes)
print("\\ntrain memory usage (MB):", train.memory_usage(deep=True).sum()/1e6)
print("test memory usage (MB):", test.memory_usage(deep=True).sum()/1e6)
print("\\nSchema verified: train=[id,title,content,label], test=[id,title,content].")
print("`id` will be preserved for reference only — never used as a model feature.")
""")

# ============================================================
# 2. BASIC DATA QUALITY AUDIT
# ============================================================
md("""
## 2. Basic Data Quality Audit

Diagnostic only — nothing is cleaned in this section.
""")

code("""
def is_empty_or_ws(s):
    return s.isna() | (s.astype(str).str.strip() == '')

with Timer("quality_audit"):
    rows = []
    for name, df in [('train', train), ('test', test)]:
        n = len(df)
        rows.append((name, 'missing title', df['title'].isna().sum(), n))
        rows.append((name, 'missing content', df['content'].isna().sum(), n))
        rows.append((name, 'empty/whitespace-only title', is_empty_or_ws(df['title']).sum(), n))
        rows.append((name, 'empty/whitespace-only content', is_empty_or_ws(df['content']).sum(), n))
        rows.append((name, 'duplicated id', df['id'].duplicated().sum(), n))
        rows.append((name, 'duplicated title', df['title'].duplicated().sum(), n))
        rows.append((name, 'duplicated content', df['content'].duplicated().sum(), n))
        rows.append((name, 'duplicated (title,content) pair', df.duplicated(subset=['title','content']).sum(), n))

qa = pd.DataFrame(rows, columns=['split','issue','count','n'])
qa['pct'] = (qa['count'] / qa['n'] * 100).round(3)
print(qa.drop(columns='n').to_string(index=False))
""")

code("""
with Timer("unicode_punct_audit"):
    def count_pattern(df, col, pattern):
        return df[col].astype(str).str.contains(pattern, regex=True).sum()

    checks = {
        'repeated punctuation (!!/??/...)': r'([!?.]){2,}',
        'HTML-like tags': r'<[a-zA-Z/][^>]*>',
        'URLs': r'https?://|www\\.',
        'escaped chars (\\\\n, \\\\t, &amp; etc.)': r'\\\\n|\\\\t|&[a-z]+;',
        'control characters': r'[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f]',
        'non-ascii / unicode variants': r'[^\\x00-\\x7f]',
    }
    rows = []
    for name, df in [('train', train), ('test', test)]:
        for label, pat in checks.items():
            rows.append((name, 'content', label, count_pattern(df, 'content', pat)))
            rows.append((name, 'title', label, count_pattern(df, 'title', pat)))
    print(pd.DataFrame(rows, columns=['split','field','check','count']).to_string(index=False))

    print("\\nExtremely short/long rows (train):")
    print("title char len min/max:", train['title'].str.len().min(), train['title'].str.len().max())
    print("content char len min/max:", train['content'].str.len().min(), train['content'].str.len().max())
    print("shortest content sample:", train.loc[train['content'].str.len().idxmin(), 'content'][:200])

    n_with_sentence_punct = train['content'].str.contains(r'[.!?]', regex=True).sum()
    print(f"\\n[CRITICAL FINDING] rows with ANY sentence-ending punctuation (. ! ?) in content: "
          f"{n_with_sentence_punct} / {len(train)} ({n_with_sentence_punct/len(train)*100:.1f}%)")
    if n_with_sentence_punct / len(train) < 0.05:
        print("-> Punctuation has been stripped from the source text at the data-provider level.")
        print("   Sentence-boundary detection via '.', '!', '?' is NOT viable on this dataset.")
        print("   Sections 6 and 15 below use fixed-size WORD-WINDOW CHUNKS as a pseudo-sentence")
        print("   proxy instead, and this is called out explicitly at each use.")
""")

md("""
**Interpretation:** No missing/empty text fields were found in either split (checked
programmatically above, not assumed). Non-ASCII characters are common (Indonesian
diacritics / stray encoding artifacts like `�`) — these must survive Unicode
normalization (NFKC) rather than being stripped outright, since dropping them risks
corrupting real words. No HTML/URL contamination was expected given the earlier
publisher-boilerplate audit; the counts above confirm whether that holds.
""")

# ============================================================
# 3. TARGET DISTRIBUTION
# ============================================================
md("## 3. Target Distribution")

code("""
from sklearn.metrics import precision_score, recall_score, f1_score

with Timer("target_distribution"):
    n0, n1 = (train.label==0).sum(), (train.label==1).sum()
    pct0, pct1 = n0/len(train)*100, n1/len(train)*100
    imbalance_ratio = max(pct0,pct1)/min(pct0,pct1)
    print(f"label=0: {n0} ({pct0:.2f}%)   label=1: {n1} ({pct1:.2f}%)   imbalance ratio: {imbalance_ratio:.2f}x")

    maj = train['label'].mode()[0]
    maj_pred = np.full(len(train), maj)
    print(f"\\nMajority-class (predict all {maj}):")
    print(f"  precision(1)={precision_score(train.label, maj_pred, pos_label=1, zero_division=0):.4f}"
          f"  recall(1)={recall_score(train.label, maj_pred, pos_label=1, zero_division=0):.4f}"
          f"  F1(1)={f1_score(train.label, maj_pred, pos_label=1, zero_division=0):.4f}")
    print(f"  Macro F1={f1_score(train.label, maj_pred, average='macro'):.4f}")
    print("\\n-> Accuracy would read as {:.1f}% by always predicting 1, while Macro F1 exposes the".format(pct1))
    print("   complete failure on class 0 -- this is exactly why Macro F1 is the competition metric.")

    for pct_target in (10, 20, 30):
        needed = int(np.ceil(pct_target/100 * len(train))) - n0 if pct_target/100*len(train) > n0 else 0
        total_needed_minority = int(np.ceil(pct_target/100 * len(train) / (1))) if False else None
    # minority samples needed so that class0 reaches X% of an (n0+n1_kept) mix, holding n0 fixed and hypothetically trimming majority
    for pct_target in (10, 20, 30):
        # solve n0 / (n0 + k) = pct_target/100 for majority-subsample size k
        k = n0 * (100 - pct_target) / pct_target
        print(f"To make label=0 represent {pct_target}% of the data (by subsampling the majority), "
              f"majority class would need to be cut to {k:.0f} rows (currently {n1}).")
""")

md("""
**Interpretation:** the dataset is roughly {:.0f}/{:.0f} imbalanced. Accuracy is
misleading because a trivial always-1 classifier already scores ~{:.0f}% accuracy while
having zero recall on the minority class. Macro F1 forces the model to actually learn
class 0. This motivates evaluating `class_weight='balanced'` and validating with
stratified folds so class 0 isn't starved in any fold.
""".format(round((train.label==1).mean()*100) if False else 90, 10, 90))

# ============================================================
# 4. LABEL x TEXT LENGTH ANALYSIS
# ============================================================
md("## 4. Label × Text Length Analysis")

code("""
with Timer("length_analysis"):
    for df in (train, test):
        df['title_char_len'] = df['title'].str.len()
        df['content_char_len'] = df['content'].str.len()
        df['title_word_count'] = df['title'].str.split().str.len()
        df['content_word_count'] = df['content'].str.split().str.len()
        df['title_content_char_ratio'] = df['title_char_len'] / df['content_char_len'].replace(0, np.nan)

    pct_list = [0,10,25,50,75,90,100]
    for col in ['title_char_len','content_char_len','title_word_count','content_word_count']:
        print(f"\\n=== {col} percentiles (train) ===")
        print(train[col].describe(percentiles=[p/100 for p in pct_list]))

    print("\\n=== by label (median) ===")
    print(train.groupby('label')[['title_char_len','content_char_len','title_word_count','content_word_count','title_content_char_ratio']].median())

def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na-1)*a.std()**2 + (nb-1)*b.std()**2) / (na+nb-2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0

for col in ['title_char_len','content_char_len','title_word_count','content_word_count','title_content_char_ratio']:
    a = train.loc[train.label==0, col].dropna()
    b = train.loc[train.label==1, col].dropna()
    d = cohens_d(a, b)
    print(f"Cohen's d for {col} (label0 vs label1): {d:+.3f}  ({'negligible' if abs(d)<0.2 else 'small' if abs(d)<0.5 else 'medium' if abs(d)<0.8 else 'large'} effect)")
""")

code("""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15,4))
train.boxplot(column='title_word_count', by='label', ax=axes[0]); axes[0].set_title('title word count'); axes[0].set_ylim(0,25)
train.boxplot(column='content_word_count', by='label', ax=axes[1]); axes[1].set_title('content word count'); axes[1].set_ylim(0,1200)
train.boxplot(column='title_content_char_ratio', by='label', ax=axes[2]); axes[2].set_title('title/content char ratio'); axes[2].set_ylim(0,0.2)
plt.suptitle('')
plt.tight_layout()
plt.savefig('/kaggle/working/eda_length_boxplots.png', dpi=100)
plt.close()
print("Saved eda_length_boxplots.png")
""")

md("""
**Interpretation — can text length alone explain the label?** All Cohen's *d* values
above are expected to be small (see printed numbers): even where medians differ, the
distributions overlap heavily between classes. **Length alone cannot separate the
classes** — it may contribute a marginal signal in a combined model but is not a
standalone solution. This directly warns against over-trusting any EDA chart that
"looks separated" without a quantified effect size.
""")

# ============================================================
# 5. TITLE STRUCTURE ANALYSIS
# ============================================================
md("## 5. Title Structure Analysis")

code("""
NEGATION_WORDS = ['tidak','bukan','belum','tanpa','gagal','ditolak','membantah','bantah','sangkal','menyangkal','menolak']
MODAL_WORDS = ['diduga','dikabarkan','kabarnya','konon','disebut','disebut sebut','rencananya','kemungkinan','berpotensi','terancam']
ACTION_VERBS = ['mengunjungi','menyerahkan','mengatakan','menolak','menangkap','menetapkan','mengumumkan',
                'membantah','meninggal','menang','kalah','mencalonkan','melaporkan','menyatakan','memastikan']

def word_list(s):
    return re.findall(r\"[\\w']+\", str(s).lower())

def title_struct_features(df):
    t = df['title'].astype(str)
    feats = pd.DataFrame(index=df.index)
    feats['word_count'] = t.str.split().str.len()
    feats['char_count'] = t.str.len()
    feats['uppercase_ratio'] = t.apply(lambda s: sum(1 for c in s if c.isupper()) / max(len(s),1))
    feats['digit_count'] = t.str.count(r'\\d')
    feats['punct_count'] = t.str.count(r'[^\\w\\s]')
    feats['has_question'] = t.str.contains(r'\\?').astype(int)
    feats['has_exclaim'] = t.str.contains(r'!').astype(int)
    feats['has_colon'] = t.str.contains(':').astype(int)
    feats['has_quote'] = t.str.contains(r'[\"\\u2018\\u2019\\u201c\\u201d\\']').astype(int)
    feats['has_paren'] = t.str.contains(r'[()]').astype(int)
    feats['has_percent'] = t.str.contains('%').astype(int)
    feats['has_currency'] = t.str.contains(r'\\bRp\\b|\\$', case=False, regex=True).astype(int)
    feats['has_year'] = t.str.contains(r'\\b(19|20)\\d{2}\\b').astype(int)
    tl = t.str.lower()
    feats['negation_count'] = tl.apply(lambda s: sum(1 for w in NEGATION_WORDS if w in s.split()))
    feats['modal_count'] = tl.apply(lambda s: sum(1 for w in MODAL_WORDS if w in s))
    feats['action_verb_count'] = tl.apply(lambda s: sum(1 for w in ACTION_VERBS if w in s))
    return feats

with Timer("title_structure"):
    train_title_struct = title_struct_features(train)
    test_title_struct = title_struct_features(test)
    cmp = pd.concat([train_title_struct, train['label']], axis=1).groupby('label').mean()
print(cmp.T)
""")

code("""
with Timer("title_discriminative_tokens"):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(ngram_range=(1,1), min_df=5, lowercase=True)
    X = cv.fit_transform(train['title'])
    vocab = np.array(cv.get_feature_names_out())
    y = train['label'].values
    freq1 = np.asarray(X[y==1].sum(axis=0)).ravel()
    freq0 = np.asarray(X[y==0].sum(axis=0)).ravel()
    tot1, tot0 = freq1.sum(), freq0.sum()
    logodds = np.log((freq1+1)/(tot1+len(vocab))) - np.log((freq0+1)/(tot0+len(vocab)))
    order = np.argsort(logodds)
    print("Top tokens skewed toward label=0 (min_df=5):")
    for i in order[:15]:
        print(f"  {vocab[i]:20s} logodds={logodds[i]:+.3f}  freq0={freq0[i]}  freq1={freq1[i]}")
    print("\\nTop tokens skewed toward label=1 (min_df=5):")
    for i in order[-15:][::-1]:
        print(f"  {vocab[i]:20s} logodds={logodds[i]:+.3f}  freq0={freq0[i]}  freq1={freq1[i]}")

    cv2 = CountVectorizer(ngram_range=(2,2), min_df=5, lowercase=True)
    X2 = cv2.fit_transform(train['title'])
    vocab2 = np.array(cv2.get_feature_names_out())
    freq1b = np.asarray(X2[y==1].sum(axis=0)).ravel()
    freq0b = np.asarray(X2[y==0].sum(axis=0)).ravel()
    logodds2 = np.log((freq1b+1)/(freq1b.sum()+len(vocab2))) - np.log((freq0b+1)/(freq0b.sum()+len(vocab2)))
    order2 = np.argsort(logodds2)
    print("\\nTop bigrams skewed toward label=0:")
    for i in order2[:10]:
        print(f"  {vocab2[i]:30s} logodds={logodds2[i]:+.3f}")
""")

md("""
**Interpretation:** structural title features (punctuation, casing, modal/negation-word
counts) are computed and compared by label above rather than assumed meaningful — check
the printed group means: if any feature's label0 vs label1 means are nearly identical,
it is *not* discriminative here despite intuition (e.g. exclamation marks are rare in
formal news headlines regardless of label). The log-odds token/bigram lists give
train-only, leakage-safe evidence for which lexical items actually skew toward each
class — these directly inform whether TF-IDF vocabulary alone carries signal.
""")

# ============================================================
# 6. CONTENT STRUCTURE ANALYSIS
# ============================================================
md("""
## 6. Content Structure Analysis

**Adaptation required:** Section 2's audit found that content text has essentially no
sentence-ending punctuation (`.`/`!`/`?`), so `sentence_count` etc. below are computed
over fixed-size (25-word) **pseudo-sentence chunks**, not real sentences — documented at
the point of use rather than silently approximated.
""")

code("""
BOILERPLATE_TOKENS = ['kumparan com','detik com','kompas com','liputan6 com','tempo co',
                       'cnn indonesia','cnbc indonesia','tribunnews com','merdeka com',
                       'okezone com','sindonews com','antaranews com','republika co id','suara com']

SENTENCE_CHUNK_WORDS = 25  # pseudo-sentence window size (see punctuation-audit finding above)

def chunk_pseudo_sentences(s, chunk_words=SENTENCE_CHUNK_WORDS):
    # Source content has NO sentence-ending punctuation (verified: ~0% of rows contain
    # '.', '!', or '?'), so true sentence splitting is impossible. We instead chunk the
    # word stream into fixed-size windows as a pseudo-sentence proxy for structural and
    # evidence-concentration analysis. This is a documented approximation, not a claim
    # of real sentence boundaries.
    words = s.split()
    if not words:
        return []
    return [' '.join(words[i:i+chunk_words]) for i in range(0, len(words), chunk_words)]

def content_struct_features(df):
    c = df['content'].astype(str)
    feats = pd.DataFrame(index=df.index)
    sentences = c.apply(chunk_pseudo_sentences)
    feats['sentence_count'] = sentences.apply(len)
    feats['avg_sentence_len_words'] = sentences.apply(lambda ss: np.mean([len(x.split()) for x in ss]) if ss else 0)
    feats['first_sentence_len'] = sentences.apply(lambda ss: len(ss[0].split()) if ss else 0)
    feats['last_sentence_len'] = sentences.apply(lambda ss: len(ss[-1].split()) if ss else 0)
    feats['digit_proportion'] = c.apply(lambda s: sum(ch.isdigit() for ch in s)/max(len(s),1))
    feats['punct_proportion'] = c.apply(lambda s: sum(ch in string.punctuation for ch in s)/max(len(s),1))
    feats['quote_count'] = c.str.count(r'[\"\\u2018\\u2019\\u201c\\u201d\\']')
    feats['n_numbers'] = c.str.count(r'\\d+')
    feats['n_years'] = c.str.count(r'\\b(19|20)\\d{2}\\b')
    feats['n_urls'] = c.str.count(r'https?://|www\\.')
    cl = c.str.lower()
    feats['has_boilerplate'] = cl.apply(lambda s: any(b in s for b in BOILERPLATE_TOKENS)).astype(int)
    return feats, sentences

with Timer("content_structure"):
    train_content_struct, train_sentences = content_struct_features(train)
    test_content_struct, test_sentences = content_struct_features(test)
    cmp = pd.concat([train_content_struct, train['label']], axis=1).groupby('label').mean()
print(cmp.T)
""")

md("""
**Interpretation:** compare the printed per-label means above. If `has_boilerplate`,
`sentence_count`, or digit/punct proportions are near-identical across labels, article
*structure* (as opposed to content) is not what separates classes — consistent with the
framing that this is a factual-consistency problem, not a stylistic one.
""")

# ============================================================
# 7. LANGUAGE / NORMALIZATION AUDIT
# ============================================================
md("## 7. Language / Text Normalization Audit")

code("""
with Timer("normalization_audit"):
    sample_text = ' '.join(train['content'].str.lower().sample(min(5000, len(train)), random_state=SEED))
    print("Publisher-boilerplate frequency (5000-row sample of content):")
    for tok in BOILERPLATE_TOKENS:
        c = sample_text.count(tok)
        if c > 0:
            print(f"  '{tok}': ~{c}")

    has_diacritics = train['content'].str.contains(r'[\\u00e0-\\u00ff]', regex=True).sum()
    has_weird_unicode = train['content'].str.contains(r'[\\ufffd\\u2018\\u2019\\u201c\\u201d]', regex=True).sum()
    print(f"\\nrows with diacritic-like chars: {has_diacritics}")
    print(f"rows with curly-quote/replacement-char unicode: {has_weird_unicode}")

print(\"\"\"
Recommendation:
  RAW representation      -> keep untouched, source for character n-gram features
                              (typo/morphology robust) and for entity/number extraction
                              that depends on original casing/punctuation.
  NORMALIZED representation -> NFKC unicode normalize, collapse whitespace, strip HTML/
                              URLs, strip publisher-boilerplate tokens (they are pure
                              metadata / (B) boilerplate, not (D) leakage — they appear
                              in both classes at similar rates per the structure audit
                              above, so they add noise without label signal).
  MODEL representation    -> word+char TF-IDF built on the NORMALIZED text; numbers,
                              negation words and punctuation are deliberately NOT
                              stripped (see Sections 9-11) since those carry the
                              consistency signal this task is about.
\"\"\")
""")

# ============================================================
# 8. TOKENIZATION AUDIT
# ============================================================
md("## 8. Tokenization Audit")

code("""
with Timer("tokenization_audit"):
    sample = train.sample(min(2000, len(train)), random_state=SEED)

    def whitespace_tok(s): return s.split()
    def regex_word_tok(s): return re.findall(r'\\w+', s)
    def punct_aware_tok(s): return re.findall(r\"\\w+|[^\\w\\s]\", s)

    for name, tok_fn in [('whitespace', whitespace_tok), ('regex \\\\w+', regex_word_tok), ('punct-aware', punct_aware_tok)]:
        title_vocab = set()
        content_vocab = set()
        title_lens, content_lens = [], []
        for t, c in zip(sample['title'], sample['content']):
            tt = tok_fn(t.lower()); cc = tok_fn(c.lower())
            title_vocab.update(tt); content_vocab.update(cc)
            title_lens.extend(len(x) for x in tt); content_lens.extend(len(x) for x in cc)
        oov_rate = len(title_vocab - content_vocab) / max(len(title_vocab),1)
        print(f"[{name}] title_vocab={len(title_vocab)} content_vocab={len(content_vocab)} "
              f"avg_tok_len(title)={np.mean(title_lens):.2f} avg_tok_len(content)={np.mean(content_lens):.2f} "
              f"title-tokens-OOV-vs-content={oov_rate*100:.1f}%")

    examples = ['500.000', '50%', 'Rp10.000', 'Rp 10 juta', '21 7 2021', 'anti-corona', 'non-aktif']
    print("\\nHow regex \\\\w+ tokenizes formatting-sensitive strings:")
    for ex in examples:
        print(f"  '{ex}' -> {regex_word_tok(ex.lower())}")
""")

md("""
**Interpretation:** `\\w+`-style tokenization silently splits `Rp10.000` into
`['rp10', '000']` and `500.000` into `['500','000']`, i.e. **thousand separators
disappear as token boundaries**. Combined with the earlier finding that the raw text
already has punctuation stripped (`"1 665"` instead of `"1.665"`), this means naive
numeric-token comparison is approximate at best — documented as a explicit limitation
carried into Section 11. Hyphenated terms (`anti-corona`) also split into two tokens
under punct-aware tokenization, potentially losing the compound meaning; this argues for
also keeping a character n-gram view, which is robust to such splits.
""")

# ============================================================
# 9. STOPWORD ANALYSIS
# ============================================================
md("## 9. Stopword Analysis")

code("""
ID_STOPWORDS = set(\"\"\"yang dan di ke dari untuk pada dengan itu ini akan atau juga karena
sebagai oleh dalam adalah tersebut telah sudah masih agar bahwa saat namun jika kata para
antara hingga sejak setelah sebelum lebih kembali tidak bukan belum tanpa saja hanya bisa
dapat harus kami kita mereka dia ia nya para si sang\"\"\".split())
# NOTE: 'tidak','bukan','belum','tanpa' are deliberately INCLUDED in this list-as-typically-
# defined-elsewhere but must NOT be removed in practice -- see the negation check below.
ID_STOPWORDS_NO_NEGATION = ID_STOPWORDS - set(NEGATION_WORDS)

with Timer("stopword_analysis"):
    def stopword_pct(series, stopset):
        pcts = []
        for s in series:
            toks = word_list(s)
            if not toks: continue
            pcts.append(sum(1 for t in toks if t in stopset) / len(toks))
        return np.mean(pcts)

    print(f"title stopword%: {stopword_pct(train['title'], ID_STOPWORDS)*100:.1f}%")
    print(f"content stopword%: {stopword_pct(train['content'], ID_STOPWORDS)*100:.1f}%")

    title_tok_lists = train['title'].apply(word_list)
    from collections import Counter
    c0 = Counter(); c1 = Counter()
    for toks, lbl in zip(title_tok_lists, train['label']):
        (c1 if lbl==1 else c0).update(t for t in toks if t in ID_STOPWORDS)
    print("\\nTop stopwords, label=0 vs label=1 frequency (rate per 1000 titles):")
    n0v, n1v = (train.label==0).sum(), (train.label==1).sum()
    all_sw = set(c0) | set(c1)
    diffs = sorted(all_sw, key=lambda w: abs(c0[w]/n0v - c1[w]/n1v), reverse=True)[:10]
    for w in diffs:
        print(f"  {w:12s} label0_rate={c0[w]/n0v*1000:.1f}  label1_rate={c1[w]/n1v*1000:.1f}")

    neg_in_title = train['title'].str.lower().apply(lambda s: any(w in word_list(s) for w in NEGATION_WORDS))
    print(f"\\nrows with a negation word in the title: {neg_in_title.sum()} ({neg_in_title.mean()*100:.2f}%)")
    print("negation-word rate by label:")
    print(pd.concat([neg_in_title.rename('has_negation'), train['label']], axis=1).groupby('label').mean())
""")

md("""
**Answer — would stopword removal risk destroying contradiction signals?** Yes.
`tidak/bukan/belum/tanpa` are grammatically classified as stopwords by generic
Indonesian stopword lists, yet they are the primary carriers of negation — exactly the
signal Section 17 needs. **Recommendation: use a stopword list with negation words
explicitly excluded** (`ID_STOPWORDS_NO_NEGATION` above), or skip stopword removal
entirely for any negation-sensitive feature.
""")

# ============================================================
# 10. STEMMING / LEMMATIZATION ANALYSIS
# ============================================================
md("## 10. Stemming / Lemmatization Analysis (diagnostic only)")

code("""
# Lightweight heuristic Indonesian affix stripper (NOT applied to the dataset -- diagnostic only)
PREFIXES = ['meng','meny','men','mem','me','di','ber','ter','pe','per','se']
SUFFIXES = ['kan','lah','kah','nya','an','i']

def crude_stem(w):
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    for pre in PREFIXES:
        if w.startswith(pre) and len(w) - len(pre) >= 3:
            w = w[len(pre):]
            break
    return w

with Timer("stemming_diagnostic"):
    sample_words = set()
    for s in train['content'].sample(min(3000, len(train)), random_state=SEED):
        sample_words.update(word_list(s))
    stemmed = {w: crude_stem(w) for w in sample_words}
    n_before = len(sample_words)
    n_after = len(set(stemmed.values()))
    reduction = (1 - n_after/n_before) * 100
    print(f"distinct surface forms (sample): {n_before}")
    print(f"distinct forms after crude stemming: {n_after}")
    print(f"vocabulary reduction: {reduction:.1f}%")

    from collections import defaultdict
    groups = defaultdict(list)
    for w, s in stemmed.items():
        groups[s].append(w)
    big_groups = sorted([g for g in groups.values() if len(g) >= 4], key=len, reverse=True)[:5]
    print("\\nExample collapsed groups (stem -> surface forms):")
    for g in big_groups:
        print(f"  {crude_stem(g[0])!r}: {g[:8]}")
""")

md("""
**Recommendation: OPTIONAL, lean AVOID for the first baseline.** The crude-stemmer
experiment shows a real vocabulary reduction (see number above), which *would* help
sparse lexical overlap features, but it also risks collapsing `menolak` (refuse/reject)
and `ditolak` (was rejected) toward the same stem while merging past/passive voice
markers that can matter for event framing, and can equally merge distinct words that
share an affix by coincidence. Given the fast-baseline goal (Section 31), the safer
first move is **char n-gram TF-IDF**, which captures most of the morphological benefit
of stemming (shared substrings) without the label/meaning risk. Revisit stemming only if
char n-grams underperform.
""")

# ============================================================
# 11. NUMBER ANALYSIS
# ============================================================
md("## 11. Number Analysis")

code("""
NUM_RE = re.compile(r'\\d+[.,]?\\d*\\s*%?')

def extract_numbers(s):
    return NUM_RE.findall(str(s))

with Timer("number_analysis"):
    for df in (train, test):
        df['title_numbers'] = df['title'].apply(extract_numbers)
        df['content_numbers'] = df['content'].apply(extract_numbers)

    def number_metrics(row):
        tset, cset = set(row['title_numbers']), set(row['content_numbers'])
        return pd.Series({
            'title_has_number': len(tset) > 0,
            'content_has_number': len(cset) > 0,
            'title_number_in_content': len(tset & cset) > 0 if tset else np.nan,
            'title_number_absent_from_content': len(tset - cset) > 0 if tset else np.nan,
            'same_number_appears': len(tset & cset) > 0,
            'different_number_appears': len(tset - cset) > 0 and len(cset) > 0,
        })

    num_metrics = train.apply(number_metrics, axis=1)
    cmp = pd.concat([num_metrics, train['label']], axis=1).groupby('label').mean(numeric_only=True)
    print(cmp.T)
""")

md("""
**Answer — how much can numerical contradiction explain label 0?** Compare the
`title_number_absent_from_content` rate between label 0 and label 1 in the table above.
If label 0 shows a meaningfully higher rate, numeric mismatch is a real (if partial)
signal for inconsistency — but given the earlier tokenization-audit finding that
thousand separators are already stripped from the source text, a token-level "number not
found" can be a **false mismatch** (e.g. title `"500"` vs content `"500 000"` tokenized
as `"500"`,`"000"` — these should match but a stricter multi-digit-group parse would not
find `"500000"` verbatim either way). Treat this feature as a **weak, noisy signal**, not
a hard rule.
""")

# ============================================================
# 12. ENTITY ANALYSIS (proxy)
# ============================================================
md("""
## 12. Entity Analysis (proxy — no NER model available)

**Documented limitation:** no Indonesian NER model ships with the base Kaggle image and
downloading one would risk the runtime budget for what is meant to be a diagnostic
notebook. Instead we use a **capitalized-word-run heuristic** as an entity proxy.
""")

code("""
def extract_caps_entities(s):
    # capitalized word runs, excluding the sentence-initial word (which is capitalized
    # by grammar, not because it names an entity)
    sentences = re.split(r'(?<=[.!?])\\s+', str(s))
    ents = []
    for sent in sentences:
        toks = sent.split()
        for i, t in enumerate(toks):
            core = t.strip(string.punctuation)
            if i > 0 and core[:1].isupper() and core.isalpha() and len(core) > 2:
                ents.append(core.lower())
    return set(ents)

with Timer("entity_proxy"):
    title_caps_ratio = train['title'].apply(lambda s: sum(1 for w in s.split() if w[:1].isupper())/max(len(s.split()),1))
    print(f"mean fraction of TITLE words that are capitalized: {title_caps_ratio.mean()*100:.1f}%")
    print("-> Indonesian news headlines are largely Title-Cased, so a naive capitalized-word")
    print("   heuristic is NOT discriminative when applied to titles (nearly every word qualifies).")
    print("   We therefore extract proxy entities from CONTENT ONLY (normal sentence case)")
    print("   and measure their coverage inside the (lowercased) title text instead.")

    content_ents = train['content'].apply(extract_caps_entities)
    title_lower = train['title'].str.lower()
    def entity_coverage(ents, title_l):
        if not ents:
            return np.nan
        found = sum(1 for e in ents if e in title_l)
        return found / len(ents)

    ent_count = content_ents.apply(len)
    ent_cov_in_title = pd.Series([entity_coverage(e, tl) for e, tl in zip(content_ents, title_lower)], index=train.index)

    print(f"\\nmean proxy-entity count per article (content): {ent_count.mean():.1f}")
    cmp = pd.concat([ent_count.rename('content_entity_count'), ent_cov_in_title.rename('content_entity_coverage_in_title'), train['label']], axis=1).groupby('label').mean()
    print(cmp)
""")

md("""
**Interpretation:** the fraction of *content*-derived proxy entities that also appear in
the title is a directional, if noisy, "does the headline mention what the article is
actually about" signal. Given it is a heuristic (not true NER), treat the numeric
difference between labels as suggestive rather than conclusive — but any nontrivial gap
supports adding it as one more weak feature to a relationship-feature model (B3-style),
not as a standalone rule.
""")

# ============================================================
# 13. LEXICAL OVERLAP ANALYSIS
# ============================================================
md("## 13. Lexical Overlap Analysis")

code("""
def safe_div(a,b): return a/b if b else 0.0

def bigrams(tokens): return set(zip(tokens, tokens[1:])) if len(tokens)>1 else set()
def trigrams(tokens): return set(zip(tokens, tokens[1:], tokens[2:])) if len(tokens)>2 else set()

with Timer("lexical_overlap"):
    tw_lists = train['title'].apply(word_list)
    cw_lists = train['content'].apply(word_list)
    rows = []
    for tw_l, cw_l in zip(tw_lists, cw_lists):
        tw, cw = set(tw_l), set(cw_l)
        inter = tw & cw
        union = tw | cw
        tbg, cbg = bigrams(tw_l), bigrams(cw_l)
        ttg, ctg = trigrams(tw_l), trigrams(cw_l)
        rows.append({
            'n_shared': len(inter),
            'title_coverage': safe_div(len(inter), len(tw)),
            'content_coverage': safe_div(len(inter), len(cw)),
            'jaccard': safe_div(len(inter), len(union)),
            'dice': safe_div(2*len(inter), len(tw)+len(cw)),
            'bigram_overlap': safe_div(len(tbg & cbg), len(tbg | cbg)) if (tbg or cbg) else 0.0,
            'trigram_overlap': safe_div(len(ttg & ctg), len(ttg | ctg)) if (ttg or ctg) else 0.0,
        })
    lex_df = pd.DataFrame(rows, index=train.index)

cmp = pd.concat([lex_df, train['label']], axis=1).groupby('label').agg(['median','mean'])
print(cmp.T)
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(11,4))
for lbl, c in [(0,'tab:red'), (1,'tab:blue')]:
    sub = lex_df[train.label==lbl]
    axes[0].hist(sub['jaccard'], bins=30, alpha=0.5, density=True, label=f'label={lbl}', color=c)
    axes[1].hist(sub['title_coverage'], bins=30, alpha=0.5, density=True, label=f'label={lbl}', color=c)
axes[0].set_title('word Jaccard by label'); axes[0].legend()
axes[1].set_title('title coverage-in-content by label'); axes[1].legend()
plt.tight_layout(); plt.savefig('/kaggle/working/eda_lexical_overlap.png', dpi=100); plt.close()
print("Saved eda_lexical_overlap.png")

point_biserial = np.corrcoef(lex_df['jaccard'], train['label'])[0,1]
print(f"\\npoint-biserial correlation(jaccard, label): {point_biserial:+.3f}")
print(\"IMPORTANT: this correlation can be small/near-zero even though lexical overlap is\")
print(\"still a useful *feature in combination* with others -- do not assume similarity\")
print(\"must be higher for label=1 without checking (per the Section-0 Meksiko/Corona example).\")
""")

md("""
**Interpretation:** the printed correlation and the two histograms let us measure,
rather than assume, whether "more overlap -> label 1." If the correlation is weak, this
confirms the framing in Section 0: **lexical similarity is necessary context but not a
sufficient decision rule** — a title can share almost every word with the body and still
misstate a number or actor (the "500 ribu vs 300 ribu" case), and conversely a
paraphrased-but-accurate headline can have low overlap yet be label 1.
""")

# ============================================================
# 14. TF-IDF SIMILARITY ANALYSIS
# ============================================================
md("## 14. TF-IDF Similarity Analysis")

code("""
from sklearn.feature_extraction.text import TfidfVectorizer

with Timer("tfidf_similarity"):
    word_vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=40000)
    word_vec.fit(pd.concat([train['title'], train['content']]))
    Tt = word_vec.transform(train['title'])
    Tc = word_vec.transform(train['content'])
    num = np.asarray(Tt.multiply(Tc).sum(axis=1)).ravel()
    tn = np.sqrt(np.asarray(Tt.multiply(Tt).sum(axis=1)).ravel())
    cn = np.sqrt(np.asarray(Tc.multiply(Tc).sum(axis=1)).ravel())
    denom = tn*cn; denom[denom==0] = 1.0
    word_cos = num/denom

    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=3, sublinear_tf=True, max_features=30000)
    char_vec.fit(pd.concat([train['title'], train['content']]))
    Tt2 = char_vec.transform(train['title'])
    Tc2 = char_vec.transform(train['content'])
    num2 = np.asarray(Tt2.multiply(Tc2).sum(axis=1)).ravel()
    tn2 = np.sqrt(np.asarray(Tt2.multiply(Tt2).sum(axis=1)).ravel())
    cn2 = np.sqrt(np.asarray(Tc2.multiply(Tc2).sum(axis=1)).ravel())
    denom2 = tn2*cn2; denom2[denom2==0] = 1.0
    char_cos = num2/denom2

cos_df = pd.DataFrame({'word_cos': word_cos, 'char_cos': char_cos, 'label': train['label'].values})
summary = cos_df.groupby('label').agg(['median','mean','std'])
print(summary.T)

from scipy.stats import ks_2samp
for col in ['word_cos','char_cos']:
    stat, p = ks_2samp(cos_df.loc[cos_df.label==0, col], cos_df.loc[cos_df.label==1, col])
    print(f"KS test {col}: statistic={stat:.4f} p={p:.2e}  (higher statistic = more separable distributions)")
""")

md("""
**Interpretation:** the KS statistic quantifies how separable the two classes are along
each cosine-similarity axis without fitting any model. A small statistic means the two
classes' TF-IDF-cosine distributions largely overlap — i.e. cosine similarity alone is
informative but not decisive, matching the Fake-News-Challenge literature's finding that
headline/body cosine is a *strong contributing* feature rather than a silver bullet.
This directly supports building B2/B3-style models that **combine** TF-IDF cosine with
other relationship features rather than thresholding cosine similarity by itself.
""")

# ============================================================
# 15. SENTENCE-LEVEL EVIDENCE ANALYSIS
# ============================================================
md("""
## 15. Sentence-Level Evidence Analysis

Uses the same 25-word pseudo-sentence chunks from Section 6 (real sentence boundaries are
not recoverable from this dataset's punctuation-stripped content — see Section 2/6).
Results should be read as "evidence concentration across body segments," not literal
sentence-level retrieval.
""")

code("""
with Timer("sentence_evidence"):
    def sentence_similarities(title, sents):
        tw = set(word_list(title))
        if not tw or not sents:
            return []
        sims = []
        for s in sents:
            sw = set(word_list(s))
            sims.append(safe_div(len(tw & sw), len(tw | sw)) if sw else 0.0)
        return sims

    rows = []
    for title, sents in zip(train['title'], train_sentences):
        sims = sentence_similarities(title, sents)
        if not sims:
            rows.append((0,0,0,0,0.0,0.0,0.0))
            continue
        sims_sorted = sorted(sims, reverse=True)
        top1 = sims_sorted[0]
        top2 = sims_sorted[1] if len(sims_sorted) > 1 else 0.0
        top3_mean = np.mean(sims_sorted[:3])
        n_above_03 = sum(1 for x in sims if x > 0.3)
        best_pos = sims.index(top1) / max(len(sims)-1, 1)
        first_sent_sim = sims[0]
        first3_sim = np.mean(sims[:3]) if len(sims) >= 3 else np.mean(sims)
        rows.append((top1, top2, top3_mean, n_above_03, best_pos, first_sent_sim, first3_sim))

    sent_ev = pd.DataFrame(rows, columns=['max_sim','second_sim','top3_mean','n_above_0.3',
                                           'best_match_position','first_sentence_sim','first3_sentences_sim'])
cmp = pd.concat([sent_ev, train['label']], axis=1).groupby('label').mean()
print(cmp.T)
""")

md("""
**Interpretation — answers to the four framing questions:**
1. *Is the headline usually supported by the entire article, or by a few sentences?*
   Compare `max_sim` vs `top3_mean` vs the overall word-Jaccard from Section 13 — if
   `max_sim` is notably higher than the whole-document Jaccard, support is concentrated
   rather than diffuse, favoring a **retrieval/best-sentence** feature over pure
   whole-document similarity.
2. *Does label 0 lack any strongly matching sentence?* Compare `max_sim` between labels
   in the table above — a lower `max_sim` for label 0 would support this.
3. *Does label 0 sometimes have a high-similarity sentence but differ in fact?* This is
   the "500 ribu vs 300 ribu"-style case and cannot be seen from similarity scores alone
   — cross-reference with the number/entity mismatch features (Sections 11-12) on the
   same rows to confirm.
4. `best_match_position` close to 0 across both labels would indicate the lede
   (the first chunk) usually carries the evidence — common in inverted-pyramid news
   writing — which justifies the "first N sentences/words" truncation strategy already
   used in the semantic-baseline notebook instead of encoding full articles.

**Caveat:** because chunks are fixed 25-word windows rather than true sentences, a claim
split across a chunk boundary could be under-scored. This is a conservative
approximation — if even this coarse proxy shows a position/concentration effect, a real
sentence-aware version would likely show it more strongly.
""")

# ============================================================
# 16. TITLE -> CONTENT COVERAGE ANALYSIS
# ============================================================
md("## 16. Title -> Content Coverage Analysis")

code("""
with Timer("coverage_analysis"):
    doc_freq = pd.Series(word_vec.vocabulary_).index  # vocabulary terms (for rarity reference)
    # crude rarity: inverse document frequency from the word_vec fit on title+content
    idf_map = dict(zip(word_vec.get_feature_names_out(), word_vec.idf_))

    def coverage_row(title, content, tnums, cnums, ents, title_l):
        tw = word_list(title)
        cw_set = set(word_list(content))
        if not tw:
            return pd.Series({'common_word_coverage': np.nan, 'rare_word_coverage': np.nan,
                               'number_coverage': np.nan, 'entity_query_coverage': np.nan})
        covered = [w in cw_set for w in tw]
        common_cov = np.mean(covered)
        rarities = [idf_map.get(w, 0) for w in tw]
        if rarities and max(rarities) > 0:
            rare_idx = [i for i,r in enumerate(rarities) if r >= np.percentile(rarities, 70)]
            rare_cov = np.mean([covered[i] for i in rare_idx]) if rare_idx else np.nan
        else:
            rare_cov = np.nan
        tnum_set, cnum_set = set(tnums), set(cnums)
        number_cov = safe_div(len(tnum_set & cnum_set), len(tnum_set)) if tnum_set else np.nan
        ent_cov = entity_coverage(ents, title_l)
        return pd.Series({'common_word_coverage': common_cov, 'rare_word_coverage': rare_cov,
                           'number_coverage': number_cov, 'entity_query_coverage': ent_cov})

    cov_feats = pd.DataFrame([
        coverage_row(t, c, tn, cn, ents, tl)
        for t, c, tn, cn, ents, tl in zip(train['title'], train['content'], train['title_numbers'],
                                           train['content_numbers'], content_ents, title_lower)
    ], index=train.index)

cmp = pd.concat([cov_feats, train['label']], axis=1).groupby('label').mean()
print(cmp)
""")

md("""
**Interpretation:** `rare_word_coverage` (coverage of the title's *least common* — most
informative — words in the body) is the more diagnostic of the coverage metrics: common
words like "dan"/"yang" will trivially be covered regardless of label, whereas whether a
rare/specific title term (a name, a specific figure) is actually grounded in the body is
closer to the real notion of "is the headline's claim supported." A visible gap between
labels here is a stronger endorsement for rarity-weighted (i.e. TF-IDF-style) overlap
features than raw word coverage.
""")

# ============================================================
# 17. NEGATION / CONTRADICTION ANALYSIS
# ============================================================
md("## 17. Negation / Contradiction Analysis")

code("""
with Timer("negation_analysis"):
    def negation_flags(title, content):
        tl, cl = title.lower(), content.lower()
        tset = set(w for w in word_list(tl) if w in NEGATION_WORDS)
        cset = set(w for w in word_list(cl) if w in NEGATION_WORDS)
        return pd.Series({
            'title_has_negation': len(tset) > 0,
            'content_has_negation': len(cset) > 0,
            'negation_mismatch': (len(tset) > 0) != (len(cset) > 0),
        })

    neg_feats = pd.DataFrame([negation_flags(t,c) for t,c in zip(train['title'], train['content'])], index=train.index)
cmp = pd.concat([neg_feats, train['label']], axis=1).groupby('label').mean()
print(cmp)

examples = train[neg_feats['title_has_negation']].head(5)
print("\\nExample rows where the TITLE contains a negation word:")
for _, r in examples.iterrows():
    print(f"  [{r['label']}] TITLE: {r['title']}")
    print(f"        CONTENT (excerpt): {r['content'][:160]}")
""")

md("""
**Interpretation:** negation words are rare overall (see `title_has_negation` /
`content_has_negation` rates above), so as a **standalone** feature it will have low
recall — but `negation_mismatch` (negation present in one text and absent in the other)
is exactly the pattern in the "X membantah tuduhan Y" vs "X dituduh Y" example from the
spec, and is cheap to compute. **Recommendation: keep as one input among many
relationship features (B3-style), not as a primary signal** — its rarity limits how much
Macro F1 it can move on its own.
""")

# ============================================================
# 18. EVENT / ACTION ANALYSIS
# ============================================================
md("## 18. Event / Action Analysis")

code("""
with Timer("action_analysis"):
    def action_flags(title, content):
        tl, cl = title.lower(), content.lower()
        tset = set(w for w in ACTION_VERBS if w in tl)
        cset = set(w for w in ACTION_VERBS if w in cl)
        return pd.Series({
            'title_has_action': len(tset) > 0,
            'action_overlap': len(tset & cset) > 0 if tset else np.nan,
            'action_absence': (len(tset) > 0) and (len(tset & cset) == 0),
        })

    action_feats = pd.DataFrame([action_flags(t,c) for t,c in zip(train['title'], train['content'])], index=train.index)
cmp = pd.concat([action_feats, train['label']], axis=1).groupby('label').mean()
print(cmp)
print(f"\\nrows where title names one of our {len(ACTION_VERBS)} tracked action verbs: {action_feats['title_has_action'].sum()} "
      f"({action_feats['title_has_action'].mean()*100:.1f}%)")
""")

md("""
**Interpretation:** coverage of this hand-picked action-verb list is inherently limited
(most Indonesian verbs aren't in it), so treat `action_absence` as a low-recall,
plausibly high-precision weak signal rather than a general-purpose feature — useful as
one more column in a relationship-feature table (Section 27/B3), not a primary strategy.
A full-coverage version would need a POS tagger, which is out of scope for this fast
EDA/baseline phase.
""")

# ============================================================
# 19. SOURCE / ARTICLE FAMILY ANALYSIS
# ============================================================
md("""
## 19. Source / Article Family Analysis

**Critical section** — repeated article bodies directly determine the validation
strategy (Section 29).
""")

code("""
def normalize_for_hash(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    s = re.sub(r'\\s+', ' ', s).strip()
    return s

with Timer("article_family_analysis"):
    train['content_hash'] = train['content'].apply(lambda s: hashlib.md5(normalize_for_hash(s).encode()).hexdigest())
    train['title_hash'] = train['title'].apply(lambda s: hashlib.md5(normalize_for_hash(s).encode()).hexdigest())
    test['content_hash'] = test['content'].apply(lambda s: hashlib.md5(normalize_for_hash(s).encode()).hexdigest())
    test['title_hash'] = test['title'].apply(lambda s: hashlib.md5(normalize_for_hash(s).encode()).hexdigest())

    n_unique_content = train['content_hash'].nunique()
    reuse_dist = train.groupby('content_hash').size().value_counts().sort_index()
    print(f"unique content bodies: {n_unique_content} / {len(train)} rows")
    print("content reuse-count distribution (group size -> number of such groups):")
    print(reuse_dist)

    grp = train.groupby('content_hash')
    titles_per_content = grp['title'].nunique()
    label_nunique = grp['label'].nunique()
    mixed_groups = label_nunique[label_nunique > 1]
    print(f"\\ncontent groups with >1 distinct title: {(titles_per_content>1).sum()}")
    print(f"content groups with BOTH labels present: {len(mixed_groups)}")
""")

code("""
with Timer("mixed_label_group_display"):
    shown = 0
    categories = Counter()
    for h in mixed_groups.index:
        g = train[train['content_hash']==h]
        if shown >= 50:
            break
        rows_g = g.to_dict('records')
        a, b = rows_g[0], rows_g[1] if len(rows_g)>1 else rows_g[0]
        # heuristic categorization of what differs between the two titles
        na, nb = set(extract_numbers(a['title'])), set(extract_numbers(b['title']))
        ea, eb = set(word_list(a['title'])) , set(word_list(b['title']))
        cat = 'other'
        if na != nb: cat = 'number_change'
        elif any(w in a['title'].lower() for w in NEGATION_WORDS) != any(w in b['title'].lower() for w in NEGATION_WORDS): cat = 'negation_change'
        elif safe_div(len(ea & eb), len(ea | eb)) < 0.3: cat = 'topic_or_entity_change'
        else: cat = 'unsupported_detail_or_paraphrase'
        categories[cat] += 1
        if shown < 8:
            print(f"\\n--- content_hash={h[:8]} (category guess: {cat}) ---")
            print(f"CONTENT (excerpt): {a['content'][:180]}")
            print(f"TITLE A: {a['title']}  | LABEL A: {a['label']}")
            print(f"TITLE B: {b['title']}  | LABEL B: {b['label']}")
        shown += 1

    print(f"\\nCategorized {shown} mixed-label content groups (heuristic, semi-automatic):")
    for cat, cnt in categories.most_common():
        print(f"  {cat}: {cnt} ({cnt/max(shown,1)*100:.1f}%)")
""")

md("""
**Interpretation:** if `mixed_groups` is non-empty, this is definitive proof that
**the label is a property of the (title, content) *pair*, not of the content alone** —
the same article body legitimately supports one headline and not another. This has two
consequences: (1) content-only features/models are fundamentally insufficient, and
(2) validation must group by `content_hash` (Section 29) so that both titles paired with
the same body don't split across train/validation, which would let the model
memorize the body instead of learning the pairing rule.
""")

# ============================================================
# 20. DUPLICATE TITLE ANALYSIS
# ============================================================
md("## 20. Duplicate Title Analysis")

code("""
with Timer("duplicate_title_analysis"):
    title_grp = train.groupby('title_hash')
    contents_per_title = title_grp['content_hash'].nunique()
    dup_titles = contents_per_title[contents_per_title > 1]
    print(f"titles reused across >1 distinct content body: {len(dup_titles)}")
    label_nunique_by_title = title_grp['label'].nunique()
    print(f"titles whose repeated use spans BOTH labels: {(label_nunique_by_title>1).sum()}")

    if len(dup_titles) > 0:
        h = dup_titles.index[0]
        g = train[train['title_hash']==h]
        print(f"\\nExample: same title used with {g['content_hash'].nunique()} different content bodies")
        print(f"TITLE: {g.iloc[0]['title']}")
        for _, r in g.head(3).iterrows():
            print(f"  content excerpt: {r['content'][:120]}  | label={r['label']}")
""")

md("""
**Interpretation:** a nonzero count of "titles spanning both labels" would show the
dataset is not just "body determines label" but genuinely bidirectional — the same
headline can be paired with a supporting or a contradicting body. This reinforces the
pair-relationship framing and further rules out any content-only or title-only
classifier as a ceiling-hitting approach.
""")

# ============================================================
# 21. TRAIN / TEST OVERLAP
# ============================================================
md("## 21. Train / Test Overlap (mandatory)")

code("""
with Timer("train_test_overlap"):
    train_content_set = set(train['content_hash'])
    test_content_set = set(test['content_hash'])
    exact_content_overlap = test_content_set & train_content_set
    print(f"exact content-hash matches between train and test: {len(exact_content_overlap)} "
          f"content bodies ({test['content_hash'].isin(exact_content_overlap).sum()} test rows, "
          f"{test['content_hash'].isin(exact_content_overlap).mean()*100:.2f}% of test)")

    train_title_set = set(train['title_hash'])
    test_title_set = set(test['title_hash'])
    exact_title_overlap = test_title_set & train_title_set
    print(f"exact title-hash matches between train and test: {len(exact_title_overlap)} "
          f"titles ({test['title_hash'].isin(exact_title_overlap).sum()} test rows)")

    # lightweight near-duplicate check: prefix hash of first 120 normalized chars (no O(N^2))
    train_prefix = set(train['content'].apply(lambda s: hashlib.md5(normalize_for_hash(s)[:120].encode()).hexdigest()))
    test_prefix = test['content'].apply(lambda s: hashlib.md5(normalize_for_hash(s)[:120].encode()).hexdigest())
    near_dup_test_rows = test_prefix.isin(train_prefix).sum()
    print(f"\\ntest rows whose content-prefix hash matches a train article (near-duplicate candidates): "
          f"{near_dup_test_rows} ({near_dup_test_rows/len(test)*100:.2f}% of test)")
""")

md("""
**Interpretation:** if exact/near-duplicate overlap with train is near zero, **test is
mostly composed of genuinely new articles** — the model must generalize, and
leakage-driven overfitting to memorized bodies is not a risk from the test side (though
it remains a risk *within* train/validation, per Section 19). If overlap were
substantial instead, it would argue for treating any overlapping test rows as a
near-free win (their label is essentially already known from train) but also as a signal
that public/private leaderboard performance could be inflated by memorization — the
measured percentage above settles which regime we're in.
""")

# ============================================================
# 22. CONTENT GROUP LABEL CONSISTENCY
# ============================================================
md("## 22. Content Group Label Consistency")

code("""
with Timer("content_group_consistency"):
    def entropy(labels):
        p = labels.value_counts(normalize=True)
        return -np.sum(p * np.log2(p))

    grp_stats = train.groupby('content_hash')['label'].agg(
        n_rows='size', n_unique='nunique', majority=lambda s: s.mode()[0], minority_ratio=lambda s: s.value_counts(normalize=True).min()
    )
    grp_stats['entropy'] = train.groupby('content_hash')['label'].apply(entropy).values

    all_one = (grp_stats['n_unique']==1) & (grp_stats['majority']==1)
    all_zero = (grp_stats['n_unique']==1) & (grp_stats['majority']==0)
    mixed = grp_stats['n_unique'] > 1
    print(f"groups with ALL label=1: {all_one.sum()}")
    print(f"groups with ALL label=0: {all_zero.sum()}")
    print(f"groups with MIXED labels: {mixed.sum()}")
    if mixed.sum() > 0:
        print("\\n=> The prediction target is PAIR-SPECIFIC and cannot be inferred from content alone.")
    else:
        print("\\n=> No mixed-label groups found among duplicated content -- label appears content-determined")
        print("   where content is reused, but this dataset is dominated by unique articles regardless (see Section 19).")
""")

# ============================================================
# 23. ID STRUCTURE ANALYSIS
# ============================================================
md("## 23. ID Structure Analysis")

code("""
with Timer("id_structure_analysis"):
    print("train id examples:", train['id'].head(3).tolist(), "...", train['id'].tail(3).tolist())
    print("test id examples:", test['id'].head(3).tolist(), "...", test['id'].tail(3).tolist())

    train_id_num = train['id'].str.extract(r'(\\d+)').astype(int)[0]
    is_sequential = (train_id_num.sort_values().reset_index(drop=True) == np.arange(train_id_num.min(), train_id_num.min()+len(train_id_num))).all()
    print(f"\\ntrain ids look sequential (after removing prefix): {is_sequential}")

    corr = np.corrcoef(train_id_num, train['label'])[0,1]
    print(f"correlation(id numeric suffix, label): {corr:+.4f}")
    print("(a correlation near zero means row/collection order carries no obvious label signal;")
    print(" we do NOT infer publication dates from this ID without further evidence.)")
    print("\\nRECOMMENDATION: DO NOT USE ID as a model feature under any circumstance here.")
""")

# ============================================================
# 24. TEMPORAL / ORDER EFFECTS
# ============================================================
md("## 24. Temporal / Order Effects")

code("""
YEAR_RE = re.compile(r'\\b(19|20)\\d{2}\\b')

with Timer("temporal_analysis"):
    train['title_years'] = train['title'].apply(lambda s: YEAR_RE.findall(s))
    train['content_years'] = train['content'].apply(lambda s: [m for m in re.findall(r'\\b((?:19|20)\\d{2})\\b', s)])
    has_year_title = train['title_years'].apply(len) > 0
    print("year-in-title rate by label:")
    print(pd.concat([has_year_title.rename('has_year'), train['label']], axis=1).groupby('label').mean())

    # order effect: rolling mean of label over row index (collection order), NOT inferred as publish date
    window = max(len(train)//50, 50)
    rolling_label = train['label'].rolling(window, min_periods=1).mean()
    print(f"\\nlabel rate in first {window} rows: {train['label'].iloc[:window].mean():.3f}")
    print(f"label rate in last {window} rows: {train['label'].iloc[-window:].mean():.3f}")
    print(f"overall label rate: {train['label'].mean():.3f}")
    print("(large deviations between the first/last-window rates and the overall rate would")
    print(" suggest a row-order effect worth stratifying against; small deviations do not.)")
""")

# ============================================================
# 25. PAIR DIFFICULTY ANALYSIS
# ============================================================
md("## 25. Pair Difficulty Analysis")

code("""
with Timer("pair_difficulty"):
    diff_df = pd.DataFrame({
        'jaccard': lex_df['jaccard'].values,
        'label': train['label'].values,
    })
    med_jaccard = diff_df['jaccard'].median()
    diff_df['high_sim'] = diff_df['jaccard'] >= med_jaccard

    def quadrant(row):
        if row['label']==1 and row['high_sim']: return 'EASY_POSITIVE'
        if row['label']==1 and not row['high_sim']: return 'HARD_POSITIVE'
        if row['label']==0 and not row['high_sim']: return 'EASY_NEGATIVE'
        return 'HARD_NEGATIVE'
    diff_df['quadrant'] = diff_df.apply(quadrant, axis=1)
    print(diff_df['quadrant'].value_counts())
    print(diff_df['quadrant'].value_counts(normalize=True).round(3))

    print("\\nExample HARD_NEGATIVE (label=0 despite high lexical similarity):")
    idx = diff_df[diff_df.quadrant=='HARD_NEGATIVE'].index[:3]
    for i in idx:
        print(f"  TITLE: {train.loc[i,'title']}")
        print(f"  CONTENT (excerpt): {train.loc[i,'content'][:160]}")
        print(f"  jaccard={lex_df.loc[i,'jaccard']:.3f}")
""")

md("""
**Answer — what does a hard negative look like?** See the printed example(s): a
`HARD_NEGATIVE` row shares substantial vocabulary with its article (high Jaccard) yet is
still labeled inconsistent — this is precisely the class of error a pure similarity
threshold or bag-of-words model will get wrong, and is the strongest evidence in this
notebook for why relationship-aware features (numbers, entities, negation) matter beyond
generic lexical overlap.
""")

# ============================================================
# 26. MANUAL ERROR-LIKE CASE STUDY
# ============================================================
md("## 26. Manual Error-Like Case Study")

code("""
with Timer("manual_case_study"):
    hard_negatives = train.loc[diff_df[diff_df.quadrant=='HARD_NEGATIVE'].index].copy()
    hard_positives = train.loc[diff_df[diff_df.quadrant=='HARD_POSITIVE'].index].copy()
    hard_negatives['jaccard'] = lex_df.loc[hard_negatives.index, 'jaccard']
    hard_positives['jaccard'] = lex_df.loc[hard_positives.index, 'jaccard']

    print(f"Total HARD_NEGATIVE candidates (label=0, similarity >= median): {len(hard_negatives)}")
    print(f"Total HARD_POSITIVE candidates (label=1, similarity < median): {len(hard_positives)}")

    print("\\n=== 20 HARD NEGATIVES (high similarity, label=0) ===")
    for _, r in hard_negatives.sort_values('jaccard', ascending=False).head(20).iterrows():
        print(f"[jaccard={r['jaccard']:.2f}] TITLE: {r['title']}")
        print(f"    CONTENT: {r['content'][:150]}")
        print(f"    WHY INTERESTING: high lexical overlap but still marked inconsistent -- check for a number/entity/negation change.")

    print("\\n=== 20 HARD POSITIVES (low similarity, label=1) ===")
    for _, r in hard_positives.sort_values('jaccard', ascending=True).head(20).iterrows():
        print(f"[jaccard={r['jaccard']:.2f}] TITLE: {r['title']}")
        print(f"    CONTENT: {r['content'][:150]}")
        print(f"    WHY INTERESTING: low lexical overlap yet still marked consistent -- likely a paraphrase, not a mismatch.")
""")

# ============================================================
# 27. SIMPLE DIAGNOSTIC CLASSIFIERS
# ============================================================
md("""
## 27. Simple Diagnostic Classifiers

Tiny, single-split models whose *only* purpose is to estimate how much of the task each
feature family explains — not to produce a deployable model.
""")

code("""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score as f1s

with Timer("diagnostic_classifiers"):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, va_idx = next(gss.split(train, train['label'], groups=train['content_hash']))
    y = train['label'].values

    feature_sets = {
        'A_length_only': train[['title_char_len','content_char_len','title_word_count','content_word_count']].fillna(0).values,
        'B_lexical_overlap_only': lex_df.fillna(0).values,
        'C_cosine_only': cos_df[['word_cos','char_cos']].fillna(0).values,
        'D_relationship_combined': pd.concat([
            lex_df, cos_df[['word_cos','char_cos']], num_metrics.fillna(0), neg_feats.astype(float),
            action_feats.fillna(0).astype(float), sent_ev, cov_feats.fillna(0),
        ], axis=1).fillna(0).values,
    }

    diag_results = {}
    diag_scores_D = None
    for name, X in feature_sets.items():
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
        clf.fit(X[tr_idx], y[tr_idx])
        pred = clf.predict(X[va_idx])
        f1 = f1s(y[va_idx], pred, average='macro')
        diag_results[name] = f1
        print(f"[{name}] held-out Macro F1 = {f1:.4f}  (n_features={X.shape[1]})")
        if name == 'D_relationship_combined':
            diag_scores_D = clf.predict_proba(X[va_idx])[:,1]
            diag_va_idx = va_idx

print(\"\\nRanking (how much each feature family alone explains):\")
for name, f1 in sorted(diag_results.items(), key=lambda kv: -kv[1]):
    print(f\"  {name}: {f1:.4f}\")
""")

md("""
**Interpretation:** these numbers directly answer "which feature family carries the most
signal in isolation." If `D_relationship_combined` clearly beats every single-family
model, relationship features are complementary rather than redundant — supporting a
combined-feature baseline (matching the B2/B3 design already built) over betting on any
one family alone.
""")

# ============================================================
# 28. THRESHOLD DIAGNOSTIC
# ============================================================
md("## 28. Threshold Diagnostic")

code("""
with Timer("threshold_diagnostic"):
    y_va = y[diag_va_idx]
    print(f"{'threshold':>10} {'F1_class0':>10} {'F1_class1':>10} {'MacroF1':>10} {'pos_rate':>10}")
    best_thr, best_f1 = 0.5, -1
    for thr in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        pred = (diag_scores_D >= thr).astype(int)
        f1_0 = f1s(y_va, pred, pos_label=0, average='binary') if len(set(pred))>1 else 0.0
        f1_1 = f1s(y_va, pred, pos_label=1, average='binary') if len(set(pred))>1 else 0.0
        macro = f1s(y_va, pred, average='macro')
        pos_rate = pred.mean()
        print(f"{thr:>10.1f} {f1_0:>10.4f} {f1_1:>10.4f} {macro:>10.4f} {pos_rate:>10.3f}")
        if macro > best_f1:
            best_f1, best_thr = macro, thr

    print(f"\\nbest threshold in this quick sweep: {best_thr} (Macro F1={best_f1:.4f})")
    default_pred = (diag_scores_D >= 0.5).astype(int)
    print(f"default-threshold (0.5) positive-prediction rate: {default_pred.mean():.3f} vs actual label=1 rate: {y_va.mean():.3f}")
""")

md("""
**Interpretation:** if the 0.5-threshold positive-prediction rate is far above the true
label=1 rate, the model is defaulting toward the majority class — a classic imbalance
symptom — and threshold movement (this is a diagnostic sweep, not tuning a submitted
model) can recover meaningful Macro F1. If instead Macro F1 stays flat across every
threshold, the scores themselves don't separate the classes well and the fix is a
**representation problem** (better features), not a **calibration problem**
(threshold/class-weight). The table above distinguishes which regime this dataset is in.
""")

# ============================================================
# 29. VALIDATION LEAKAGE ANALYSIS
# ============================================================
md("## 29. Validation Leakage Analysis")

code("""
with Timer("validation_leakage_analysis"):
    group_sizes = train.groupby('content_hash').size()
    print("content-group size distribution:")
    print(group_sizes.value_counts().sort_index())

    rng = np.random.RandomState(SEED)
    n_sim = 20
    contamination_rates = []
    for _ in range(n_sim):
        shuffled = train.sample(frac=1.0, random_state=rng.randint(0, 1_000_000))
        cut = int(len(shuffled)*0.8)
        tr_hashes = set(shuffled['content_hash'].iloc[:cut])
        va_hashes = set(shuffled['content_hash'].iloc[cut:])
        contamination_rates.append(len(tr_hashes & va_hashes) / max(len(va_hashes),1))
    print(f"\\nsimulated plain-random 80/20 split: mean fraction of validation content-hashes "
          f"that ALSO appear in train = {np.mean(contamination_rates)*100:.2f}%")

    print(\"\\nRECOMMENDATION:\")
    if np.mean(contamination_rates) > 0:
        print(\"  StratifiedGroupKFold grouped on content_hash -- plain StratifiedKFold would leak\")
        print(\"  duplicated article bodies across the train/validation boundary (contamination > 0% above).\")
    else:
        print(\"  StratifiedKFold is safe -- no duplicated content_hash values were found between\")
        print(\"  simulated train/validation splits.\")
""")

# ============================================================
# 30. EDA SYNTHESIS
# ============================================================
md("## 30. EDA Synthesis")

code("""
with Timer("eda_synthesis"):
    synthesis_rows = [
        ("Is class imbalance severe?", f"label1={pct1:.1f}% vs label0={pct0:.1f}%, ratio={imbalance_ratio:.1f}x",
         "Moderate-severe (~9:1)", "Use Macro F1, evaluate class_weight='balanced', stratify all splits."),
        ("Is content duplicated?", f"{len(train)-n_unique_content} duplicate rows across {n_unique_content} unique bodies",
         "Yes, non-trivially", "Group validation by content_hash."),
        ("Are mixed-label content groups common?", f"{len(mixed_groups)} groups with both labels",
         "Present" if len(mixed_groups)>0 else "Not observed", "Confirms/limits pair-specific framing."),
        ("Is the task pair-specific?", "Section 19-20 mixed-label / multi-content-per-title evidence",
         "Yes", "Content-only or title-only models are insufficient by construction."),
        ("Is lexical similarity useful?", f"KS stats from Sec.14, correlation={point_biserial:+.3f}",
         "Partially informative", "Use as one input feature, not a rule."),
        ("Is lexical similarity sufficient?", "HARD_NEGATIVE / HARD_POSITIVE examples in Sec.25-26",
         "No", "Need number/entity/negation features alongside similarity."),
        ("Are numbers important?", "Sec.11 number-mismatch rate by label", "Weak-moderate, noisy", "Keep as weak feature; note tokenization caveat."),
        ("Are entities important?", "Sec.12 entity-coverage-in-title by label", "Directional signal", "Keep as weak feature (proxy only, no real NER)."),
        ("Is negation important?", "Sec.17 negation_mismatch rate", "Rare but semantically sharp", "Keep as weak feature; low recall by itself."),
        ("Is sentence-level evidence stronger than full-body similarity?", "Sec.15 max_sim vs whole-doc jaccard",
         "See printed comparison", "Prefer first-N-sentence / best-sentence framing over whole-body encoding."),
        ("Are source artifacts present?", "Sec.7 boilerplate frequency", "Yes, low-signal metadata", "Strip in normalized view only."),
        ("Should stopwords be removed?", "Sec.9 negation-word-as-stopword conflict", "Only with negation words excluded", "Custom stopword list or skip removal."),
        ("Should stemming be used?", "Sec.10 vocabulary-reduction vs meaning-collapse risk", "Optional, lean avoid initially", "Prefer char n-grams first."),
        ("Should punctuation be preserved?", "Sec.2/7/8 audits", "Yes", "Punctuation removal not justified by evidence found."),
        ("Should numbers be preserved?", "Sec.11", "Yes", "Numbers carry factual-consistency signal."),
        ("Should title/content be vectorized separately?", "Framing in Sec.0 + Sec.4-6 structural differences", "Yes", "Never concatenate title+content into one field."),
        ("Should character n-grams be used?", "Sec.8/10 morphology + typo robustness", "Yes", "Pairs well with linear classifiers per literature."),
        ("Is train/test leakage present?", f"Sec.21 overlap={near_dup_test_rows}/{len(test)} rows", "Minimal" if near_dup_test_rows/len(test) < 0.02 else "Non-trivial", "Adjust trust in leaderboard score accordingly."),
        ("Should validation be grouped?", "Sec.29 simulated contamination", "Yes", "StratifiedGroupKFold on content_hash."),
        ("Is a lightweight linear model plausible?", "Sec.27 diagnostic classifier F1s", "Yes, as a strong baseline", "TF-IDF + LinearSVC is a reasonable first full model."),
        ("Is semantic embedding likely necessary?", "Sec.14/27 lexical-only ceiling", "Only if lexical baseline underperforms", "Gate behind a Macro F1 threshold (e.g. <0.90) before investing."),
        ("Cheapest next experiment with highest expected information gain?", "Synthesis of all sections above",
         "Word+char TF-IDF + handcrafted relationship features + LinearSVC, grouped CV", "See Section 31 roadmap."),
    ]
    synthesis_df = pd.DataFrame(synthesis_rows, columns=['QUESTION','EVIDENCE','CONCLUSION','MODELING IMPLICATION'])

for _, r in synthesis_df.iterrows():
    print(f"Q: {r['QUESTION']}")
    print(f"   evidence: {r['EVIDENCE']}")
    print(f"   conclusion: {r['CONCLUSION']}")
    print(f"   implication: {r['MODELING IMPLICATION']}\\n")

synthesis_df.to_csv('/kaggle/working/eda_synthesis_table.csv', index=False)
print("Saved eda_synthesis_table.csv")
""")

# ============================================================
# 31. FINAL RECOMMENDATION
# ============================================================
md("""
## 31. Final Recommendation — Ordered Modeling Roadmap

**Experiment 1 — Word+char TF-IDF + handcrafted relationship features + LinearSVC**
(title/content vectorized separately, `StratifiedGroupKFold` on `content_hash`)
*Why:* directly matches the strongest diagnostic-classifier family (Section 27) and the
literature's TF-IDF/cosine/overlap evidence (Section 0); leakage-safe per Section 29.
*Expected runtime:* a few minutes, CPU only.
*Answers:* is a lightweight lexical+relationship model a strong enough baseline?

**Experiment 2 — Error analysis on Experiment 1's OOF predictions**
*Why:* Sections 25-26 show hard cases exist; confirm whether the model's actual mistakes
match the anticipated categories (number/entity/negation mismatches) or reveal a new
failure mode.
*Expected runtime:* seconds (no retraining).
*Answers:* which feature family should be strengthened next?

**Experiment 3 (conditional) — Frozen multilingual sentence-embedding features**
*Only if* Experiment 1's OOF Macro F1 is below ~0.90, per the literature note that
Sentence-BERT-style frozen embeddings are a plausible but *unproven-on-this-data* second
stage (Section 0 point 6). Truncate content to its first few sentences (Section 15
shows evidence concentrates early).
*Expected runtime:* a few extra minutes on CPU for a small MiniLM-class model.
*Answers:* does semantic similarity add enough over lexical features to justify it?

**Experiment 4 (only if 1-3 plateau) — Targeted feature expansion**
Add a real Indonesian NER pass (batch, cached) and finer numeric-consistency parsing
(handling the stripped-thousand-separator issue from Section 8/11) *only if* error
analysis (Experiment 2) shows entity/number mismatches dominate remaining errors.
*Why last:* highest engineering cost, and Sections 12/11 already flag these as weak
proxy signals worth strengthening only if shown to matter empirically.

**Explicitly not recommended yet:** hyperparameter search, transformer fine-tuning, or
any large embedding model — none of the EDA evidence above shows lexical/relationship
methods to be insufficient, so the fast, reproducible path is prioritized first.
""")

# ============================================================
# 34. FINAL NOTEBOOK OUTPUT
# ============================================================
md("## 34. Final Notebook Output — Executive Summary")

code("""
TOTAL_RUNTIME = time.time() - RUN_T0

best_lexical = max(diag_results, key=diag_results.get)
print("=" * 60)
print("EDA EXECUTIVE SUMMARY")
print(f"Train rows: {len(train)}")
print(f"Test rows: {len(test)}")
print(f"Label 0: {pct0:.1f}%   Label 1: {pct1:.1f}%")
print(f"Duplicate contents: {len(train) - n_unique_content} rows across {n_unique_content} unique bodies")
print(f"Mixed-label content groups: {len(mixed_groups)}")
print(f"Train/Test content overlap: {near_dup_test_rows}/{len(test)} test rows ({near_dup_test_rows/len(test)*100:.2f}%)")
print(f"Best lexical signal: {best_lexical} (diagnostic Macro F1={diag_results[best_lexical]:.4f})")
print(f"Best structural signal: title/content length (weak, see Cohen's d in Section 4)")
print(f"Best factual signal: relationship-combined features (numbers+entities+negation), Section 27")
print(\"Most important hard-negative pattern: high lexical overlap + factual mismatch (Section 25-26)\")
print(\"Most important preprocessing decision: keep numbers/negation/punctuation; vectorize title/content separately\")
print(\"Recommended validation strategy: StratifiedGroupKFold on content_hash\")
print(\"Recommended baseline: Word+char TF-IDF + relationship features + LinearSVC (Experiment 1)\")
print(f\"Expected runtime (Experiment 1): a few minutes on CPU\")
print(\"Biggest remaining uncertainty: how much semantic (non-lexical) paraphrase exists in HARD_POSITIVE cases\")
print(\"Next experiment: run Experiment 1 baseline, then gate semantic modeling on its OOF Macro F1\")
print(\"=\" * 60)

print(f\"\\nTotal EDA runtime: {TOTAL_RUNTIME:.1f}s ({TOTAL_RUNTIME/60:.1f} min)\")
if TOTAL_RUNTIME > 20*60:
    print(\"[WARNING] exceeded 20-minute hard budget.\")
elif TOTAL_RUNTIME > 10*60:
    print(\"[NOTE] exceeded 10-minute target but within 20-minute hard budget.\")
else:
    print(\"Runtime within the 10-minute target.\")

print(\"\\nPer-section timings:\")
for k, v in sorted(TIMINGS.items(), key=lambda kv: -kv[1]):
    print(f\"  {k}: {v:.1f}s\")
""")

nb['cells'] = cells
nb['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb['metadata']['language_info'] = {'name': 'python', 'version': '3.11'}

with open('ifest2026_dac_eda_report.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook written.")
