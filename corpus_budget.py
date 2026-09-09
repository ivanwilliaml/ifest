"""How much text do we actually have to pretrain a transformer from scratch?"""
import re, string, unicodedata, pandas as pd
from collections import Counter

D = r'D:\Lomba\IFEST2026_DAC\data'
tr = pd.read_csv(D + r'\train.csv'); te = pd.read_csv(D + r'\test.csv')
TC = 'title' if 'title' in tr.columns else tr.columns[1]
CC = 'content' if 'content' in tr.columns else tr.columns[2]
norm = lambda s: re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(s))).strip()

bodies, titles = [], []
for df in (tr, te):
    bodies += [norm(x) for x in df[CC]]
    titles += [norm(x) for x in df[TC]]
ub = set(bodies)
tok = lambda s: s.split()

w_all = sum(len(tok(b)) for b in bodies) + sum(len(tok(t)) for t in titles)
w_uni = sum(len(tok(b)) for b in ub) + sum(len(tok(t)) for t in set(titles))
vocab = Counter(w.lower().strip(string.punctuation) for b in ub for w in tok(b))

print('rows            : %d train + %d test' % (len(tr), len(te)))
print('unique bodies   : %d' % len(ub))
print('words (with dup): %,d'.replace(',', '') % w_all)
print('words (dedup)   : {:,}'.format(w_uni))
print('vocab types     : {:,}'.format(len(vocab)))
print('types seen >=5  : {:,}'.format(sum(1 for c in vocab.values() if c >= 5)))
print('median body len : %d words' % pd.Series([len(tok(b)) for b in ub]).median())

print('\n--- pretraining budget vs what BERT-class models actually use ---')
for name, w in [('this competition corpus (dedup)', w_uni),
                ('IndoBERT / Indo4B', 4_000_000_000),
                ('BERT-base (Wiki+Books)', 3_300_000_000),
                ('smallest usable from-scratch LM (rough)', 100_000_000)]:
    print('  %-40s {:>15,}'.format(w).replace('{:>15,}', '') % name, '{:>15,}'.format(w))
print('\nratio to IndoBERT: 1 : {:,.0f}'.format(4_000_000_000 / max(w_uni, 1)))

# the supervised budget is the real constraint
tr['ch'] = tr[CC].map(norm).str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True).map(hash)
sizes = tr.groupby('ch').size()
cold = tr[tr['ch'].map(sizes) == 1]
print('\n--- supervised budget for the thing we must learn ---')
print('labelled pairs        : {:,}'.format(len(tr)))
print('negatives (all)       : {:,}'.format(int((tr.label == 0).sum())))
print('COLD negatives        : {:,}   <- the only examples of the hard case'
      .format(int((cold.label == 0).sum())))
print('  of which invisible to entity-missing (~42%%): ~{:,}'
      .format(int((cold.label == 0).sum() * 0.42)))
