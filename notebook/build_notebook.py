import nbformat as nbf
import textwrap

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(src).strip()))

def code(src):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(src).strip()))

md("""
# Penyisihan IFEST 2026 DAC — Judul vs Isi Berita Classifier

Binary classification: apakah judul berita **sesuai** (1) atau **tidak sesuai** (0) dengan isi beritanya.
Metric: Macro F1-Score.

Pipeline: Load -> EDA -> Preprocessing -> Modeling (IndoBERT fine-tune + TF-IDF baseline) -> Inference -> Submission.
""")

code("""
import os, re, json, glob, random, time
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import torch
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
""")

code("""
# Locate data files robustly (API-pushed competitions can mount one level deeper)
DATA_FILES = {}
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f in ('train.csv', 'test.csv', 'sample_submission.csv'):
            DATA_FILES[f] = os.path.join(root, f)
print(DATA_FILES)
assert len(DATA_FILES) == 3, f"Missing data files, found: {DATA_FILES}"
""")

code("""
train = pd.read_csv(DATA_FILES['train.csv'])
test = pd.read_csv(DATA_FILES['test.csv'])
sample_sub = pd.read_csv(DATA_FILES['sample_submission.csv'])

print(train.shape, test.shape, sample_sub.shape)
train.head()
""")

md("## EDA")

code("""
print("Label distribution (train):")
print(train['label'].value_counts())
print(train['label'].value_counts(normalize=True))

print("\\nMissing values:")
print(train.isna().sum())
print(test.isna().sum())

train['title_len'] = train['title'].str.len()
train['content_len'] = train['content'].str.len()
print("\\nTitle length stats:")
print(train['title_len'].describe())
print("\\nContent length stats:")
print(train['content_len'].describe())
""")

code("""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
train['label'].value_counts().plot(kind='bar', ax=axes[0], title='Label distribution')
train['title_len'].hist(bins=50, ax=axes[1])
axes[1].set_title('Title length (chars)')
train['content_len'].clip(upper=8000).hist(bins=50, ax=axes[2])
axes[2].set_title('Content length (chars, clipped 8000)')
plt.tight_layout()
plt.savefig('/kaggle/working/eda.png', dpi=100)
print("Saved EDA plot.")
""")

code("""
# Sample of each class
pd.set_option('display.max_colwidth', 150)
print("=== Label 1 (Sesuai) samples ===")
print(train[train.label==1][['title','content']].sample(2, random_state=SEED))
print("\\n=== Label 0 (Tidak Sesuai) samples ===")
print(train[train.label==0][['title','content']].sample(2, random_state=SEED))
""")

md("## Preprocessing")

code("""
def clean_text(s):
    s = str(s)
    s = re.sub(r'\\s+', ' ', s).strip()
    return s

for df in (train, test):
    df['title'] = df['title'].apply(clean_text)
    df['content'] = df['content'].apply(clean_text)

# Truncate content by words to keep tokenization fast & within model max length
def truncate_words(s, n_words=350):
    w = s.split(' ')
    return ' '.join(w[:n_words])

train['content_trunc'] = train['content'].apply(truncate_words)
test['content_trunc'] = test['content'].apply(truncate_words)
""")

code("""
from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    np.arange(len(train)), test_size=0.12, random_state=SEED, stratify=train['label']
)
tr_df = train.iloc[train_idx].reset_index(drop=True)
va_df = train.iloc[val_idx].reset_index(drop=True)
print(tr_df.shape, va_df.shape)
print(tr_df['label'].value_counts(normalize=True))
print(va_df['label'].value_counts(normalize=True))
""")

md("## Baseline: TF-IDF + Logistic Regression")

code("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

def make_pair_text(df):
    return (df['title'] + ' [SEP] ' + df['content_trunc']).tolist()

Xtr_text = make_pair_text(tr_df)
Xva_text = make_pair_text(va_df)
Xte_text = make_pair_text(test)

tfidf = TfidfVectorizer(max_features=60000, ngram_range=(1,2), min_df=2, sublinear_tf=True)
Xtr_tfidf = tfidf.fit_transform(Xtr_text)
Xva_tfidf = tfidf.transform(Xva_text)
Xte_tfidf = tfidf.transform(Xte_text)

clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=2.0, random_state=SEED)
clf.fit(Xtr_tfidf, tr_df['label'])

val_pred_tfidf = clf.predict(Xva_tfidf)
f1_tfidf = f1_score(va_df['label'], val_pred_tfidf, average='macro')
print("TF-IDF+LogReg macro F1 (val):", f1_tfidf)
print(classification_report(va_df['label'], val_pred_tfidf, digits=4))
""")

md("## IndoBERT fine-tuning")

code("""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LEN = 384
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class PairDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, has_label=True):
        self.titles = df['title'].tolist()
        self.contents = df['content_trunc'].tolist()
        self.has_label = has_label
        if has_label:
            self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.titles[idx], self.contents[idx],
            truncation=True, max_length=self.max_len, padding='max_length',
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.has_label:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_ds = PairDataset(tr_df, tokenizer, MAX_LEN)
val_ds = PairDataset(va_df, tokenizer, MAX_LEN)
test_ds = PairDataset(test, tokenizer, MAX_LEN, has_label=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)

print(len(train_ds), len(val_ds), len(test_ds))
""")

code("""
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# class weights to counter 90/10 imbalance, tuned for macro-F1
class_counts = tr_df['label'].value_counts().sort_index()
weights = (1.0 / class_counts)
weights = weights / weights.sum() * 2
class_weights = torch.tensor(weights.values, dtype=torch.float, device=device)
print("class weights:", class_weights)

loss_fn = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)
""")

code("""
def evaluate(loader, has_label=True):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop('labels', None)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            if has_label and labels is not None:
                all_labels.extend(labels.numpy().tolist())
    return all_preds, all_labels

best_f1 = -1
best_state = None

for epoch in range(EPOCHS):
    model.train()
    t0 = time.time()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        labels = batch.pop('labels').to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(**batch)
        loss = loss_fn(out.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    val_preds, val_labels = evaluate(val_loader)
    f1 = f1_score(val_labels, val_preds, average='macro')
    print(f"Epoch {epoch+1}/{EPOCHS} | loss={total_loss/len(train_loader):.4f} | val macro-F1={f1:.4f} | time={time.time()-t0:.1f}s")

    if f1 > best_f1:
        best_f1 = f1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

print("Best val macro-F1 (IndoBERT):", best_f1)
model.load_state_dict(best_state)
""")

code("""
val_preds_bert, val_labels_bert = evaluate(val_loader)
print("IndoBERT final val report:")
print(classification_report(val_labels_bert, val_preds_bert, digits=4))
print("TF-IDF baseline macro F1 was:", f1_tfidf)
print("IndoBERT macro F1:", f1_score(val_labels_bert, val_preds_bert, average='macro'))

USE_BERT = f1_score(val_labels_bert, val_preds_bert, average='macro') >= f1_tfidf
print("Using model for final submission:", "IndoBERT" if USE_BERT else "TF-IDF+LogReg")
""")

md("## Inference & Submission")

code("""
if USE_BERT:
    test_preds, _ = evaluate(test_loader, has_label=False)
else:
    test_preds = clf.predict(Xte_tfidf)

submission = pd.DataFrame({'id': test['id'], 'label': test_preds})
submission['label'] = submission['label'].astype(int)
assert submission.shape[0] == sample_sub.shape[0]
assert set(submission['id']) == set(sample_sub['id'])
submission = submission.set_index('id').loc[sample_sub['id']].reset_index()

submission.to_csv('/kaggle/working/submission.csv', index=False)
print(submission['label'].value_counts(normalize=True))
submission.head()
""")

nb['cells'] = cells
with open('ifest2026_dac.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook written.")
