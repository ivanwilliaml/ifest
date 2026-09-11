"""
Coverage diagnostic for C3 (phrase-level predicate matching) BEFORE any CV -- per the
requested discipline: check how many of the current no_predicate_evidence_found rows would
actually get resolved by phrase-level matching (n-gram window around title predicate vs
n-gram windows in body, matched via token-stem overlap or an existing KBBI synonym/antonym
pair inside the phrase) using ONLY resources already in the pipeline (no external corpus --
Kateglo and the Liputan6 paraphrase set were both found inaccessible).
"""
import io, os, json, re, string as _s
import numpy as np, pandas as pd

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
src = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
src = src.replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
src = src.replace("/kaggle/working/submission.csv", os.path.join(d, 'local_out', 'submission.csv').replace("\\", "\\\\"))
ns = {'__name__': '__main__'}
exec(compile(src, os.path.join(d, '_dryrun_c3cov.py'), 'exec'), ns)

train = ns['train']; Ftr = ns['Ftr']; cold_mask = ns['cold_mask']
cold_oof = ns['cold_oof']; cold_y = ns['cold_y']; A_THR = ns['A_THR']
cold_pred = (cold_oof >= A_THR).astype(int)
fp_mask = (cold_y == 0) & (cold_pred == 1)
cold_idx = np.where(cold_mask)[0]
cold_train = train.iloc[cold_idx].reset_index(drop=True)
cold_feat = Ftr.iloc[cold_idx].reset_index(drop=True)
fp_train = cold_train[fp_mask].reset_index(drop=True)
fp_feat = cold_feat[fp_mask].reset_index(drop=True)

no_pred = fp_train[fp_feat['found_predicate_chunk'] == 0].reset_index(drop=True)
print(f"no_predicate_evidence_found rows in current best model: {len(no_pred)}")

cached_stem = ns['cached_stem']; toks = ns['toks']; chunks_of = ns['chunks_of']
find_predicate_in = ns['find_predicate_in']; SYNONYM_SET = ns['SYNONYM_SET']; ANTONYM_SET = ns['ANTONYM_SET']
_s_punct = _s.punctuation
STOP = {'yang','dan','di','ke','dari','pada','akan','ini','itu','juga','dengan','untuk',
        'oleh','atau','ada','tidak','sudah','telah','saat','hari','tahun','saja','bahwa'}

def phrase_window(text, center_idx, radius=2):
    words = text.split()
    lo = max(0, center_idx - radius); hi = min(len(words), center_idx + radius + 1)
    toks_ = [w.strip(_s_punct).lower() for w in words[lo:hi]]
    return [w for w in toks_ if w and w not in STOP]

def title_predicate_phrase(title):
    ti, troot = find_predicate_in(title)
    if ti is None:
        return None, None
    return phrase_window(title, ti), troot

match_none = 0; match_stem_overlap = 0; match_synonym_pair = 0; match_antonym_pair = 0
examples = {'stem_overlap': [], 'synonym_pair': [], 'antonym_pair': [], 'none': []}

for _, r in no_pred.iterrows():
    title = r['T']; body = r['C']
    tphrase, troot = title_predicate_phrase(title)
    if tphrase is None:
        match_none += 1
        continue
    t_stems = {cached_stem(w) for w in tphrase}
    best = 'none'
    body_words = body.split()
    for i, w in enumerate(body_words):
        wl = w.strip(_s_punct).lower()
        if not wl or wl in STOP:
            continue
        broot = cached_stem(wl)
        bphrase = phrase_window(body, i)
        b_stems = {cached_stem(bw) for bw in bphrase}
        # 1. any KBBI antonym pair between the two phrase stem-sets
        if best not in ('antonym_pair',) and any((ts, bs) in ANTONYM_SET for ts in t_stems for bs in b_stems):
            best = 'antonym_pair'; break
        # 2. any KBBI synonym pair between the two phrase stem-sets
        if best == 'none' and any((ts, bs) in SYNONYM_SET for ts in t_stems for bs in b_stems):
            best = 'synonym_pair'
        # 3. plain stem overlap (excluding the predicate root itself) above a small threshold
        if best == 'none':
            ov = len(t_stems & b_stems)
            if ov >= 2:
                best = 'stem_overlap'
    if best == 'antonym_pair': match_antonym_pair += 1
    elif best == 'synonym_pair': match_synonym_pair += 1
    elif best == 'stem_overlap': match_stem_overlap += 1
    else: match_none += 1
    if len(examples[best]) < 3:
        examples[best].append((title, body[:150]))

n = len(no_pred)
print(f"\n=== Phrase-level coverage (n={n}) ===")
print(f"  antonym_pair match : {match_antonym_pair}  ({match_antonym_pair/n*100:.1f}%)")
print(f"  synonym_pair match : {match_synonym_pair}  ({match_synonym_pair/n*100:.1f}%)")
print(f"  stem_overlap match : {match_stem_overlap}  ({match_stem_overlap/n*100:.1f}%)")
print(f"  no match at all    : {match_none}  ({match_none/n*100:.1f}%)")
print(f"  TOTAL resolved     : {n - match_none}  ({(n-match_none)/n*100:.1f}%)")

for k, ex in examples.items():
    print(f"\n--- examples: {k} ---")
    for t, b in ex:
        print(f"  TITLE: {t}")
        print(f"  BODY : {b}")
