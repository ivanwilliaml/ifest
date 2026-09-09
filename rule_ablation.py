"""
Which rule actually carries the score? Ablate C, B_has_pos, B_no_pos separately.

Method: grouped split on normalized content so the held-out side contains all
four regimes naturally; build the lookups from the train side only; score the
val side under each ablation. The 'model' is stubbed at its measured cold-start
behaviour (predicts 1 almost always) so we isolate the RULES' contribution.
"""
import re, unicodedata, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

D = r'D:\Lomba\IFEST2026_DAC\data'
tr = pd.read_csv(D + r'\train.csv')
TC = 'title' if 'title' in tr.columns else tr.columns[1]
CC = 'content' if 'content' in tr.columns else tr.columns[2]
norm = lambda s: re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(s))).strip()
nh = lambda x: re.sub(r'[^a-z0-9 ]', '', unicodedata.normalize('NFKC', str(x)).lower())
tr['T'] = tr[TC].map(norm); tr['C'] = tr[CC].map(norm)
tr['th'] = tr['T'].map(nh); tr['ch'] = tr['C'].map(nh)

# a split that reproduces the TEST regime mixture: split on ROWS, not groups,
# because test genuinely shares bodies with train (37% overlap measured in EDA).
rng = np.random.RandomState(0)
idx = rng.permutation(len(tr))
cut = int(0.8 * len(tr))
A, B = tr.iloc[idx[:cut]], tr.iloc[idx[cut:]]

pair = {}
for t, c, l in zip(A['th'], A['ch'], A['label']):
    pair[(t, c)] = l
byc = {}
for c, l in zip(A['ch'], A['label']):
    byc.setdefault(c, []).append(l)

reg, rule_lab = [], []
for t, c in zip(B['th'], B['ch']):
    if (t, c) in pair:
        reg.append('C_exact'); rule_lab.append(pair[(t, c)])
    elif c in byc:
        if any(l == 1 for l in byc[c]):
            reg.append('B_has_pos'); rule_lab.append(0)
        else:
            reg.append('B_no_pos'); rule_lab.append(1)
    else:
        reg.append('A_cold'); rule_lab.append(None)
reg = np.array(reg); y = B['label'].values

print('regime mixture in held-out (target test mix: cold .636 C .257 Bp .060 Bn .046)')
for r in ['A_cold', 'C_exact', 'B_has_pos', 'B_no_pos']:
    m = reg == r
    if m.sum():
        print('  %-10s n=%5d (%.3f)  true class0 rate=%.3f  rule says %s'
              % (r, m.sum(), m.mean(), 1 - y[m].mean(),
                 {'A_cold': 'model', 'C_exact': 'copy', 'B_has_pos': '0', 'B_no_pos': '1'}[r]))

# rule accuracy where it fires
print('\nrule accuracy where it fires:')
for r in ['C_exact', 'B_has_pos', 'B_no_pos']:
    m = reg == r
    if m.sum():
        pr = np.array([rule_lab[i] for i in np.where(m)[0]])
        print('  %-10s acc=%.4f  (n=%d)' % (r, (pr == y[m]).mean(), m.sum()))

MODEL_ALL_ONES = np.ones(len(y), int)   # stub: model's real cold behaviour


def score(use):
    p = MODEL_ALL_ONES.copy()
    for i, r in enumerate(reg):
        if r in use and rule_lab[i] is not None:
            p[i] = rule_lab[i]
    return f1_score(y, p, average='macro')


ALL = {'C_exact', 'B_has_pos', 'B_no_pos'}
print('\n--- ablation (model stubbed as all-ones, so this is the RULES alone) ---')
print('  no rules at all              : %.4f' % score(set()))
print('  all rules                    : %.4f' % score(ALL))
for r in sorted(ALL):
    print('  all rules MINUS %-12s : %.4f   (delta %+.4f)'
          % (r, score(ALL - {r}), score(ALL - {r}) - score(ALL)))
for r in sorted(ALL):
    print('  ONLY %-17s       : %.4f' % (r, score({r})))
