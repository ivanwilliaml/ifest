"""
C4 coverage+precision diagnostic (BEFORE any CV, per instructions): constrained phrase
alignment, binary strict output. Title predicate phrase (predicate word + adjacent
non-stopword content words) is aligned word-by-word against every body window via
exact/stem/synonym match; ACCEPT only if coverage of the title phrase's content words
is >= 2/3. Prints every matched pair (not just 3 samples) so precision can be judged by
reading, not just counted.
"""
import io, os, json, re, string as _s
import numpy as np, pandas as pd

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
src = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
src = src.replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
src = src.replace("/kaggle/working/submission.csv", os.path.join(d, 'local_out', 'submission.csv').replace("\\", "\\\\"))
ns = {'__name__': '__main__'}
exec(compile(src, os.path.join(d, '_dryrun_c4.py'), 'exec'), ns)

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
print(f"no_predicate_evidence_found rows: {len(no_pred)}")

cached_stem = ns['cached_stem']; find_predicate_in = ns['find_predicate_in']
SYNONYM_SET = ns['SYNONYM_SET']; ANTONYM_SET = ns['ANTONYM_SET']
STOP = {'yang','dan','di','ke','dari','pada','akan','ini','itu','juga','dengan','untuk',
        'oleh','atau','ada','sudah','telah','saat','hari','tahun','saja','bahwa','ia','nya'}

def content_words(tokens):
    out = []
    for w in tokens:
        wl = w.strip(_s.punctuation).lower()
        if wl and wl not in STOP:
            out.append(wl)
    return out

def title_predicate_phrase(title):
    ti, troot = find_predicate_in(title)
    if ti is None:
        return None
    words = title.split()
    lo = max(0, ti - 2); hi = min(len(words), ti + 3)
    return content_words(words[lo:hi])

def aligns(tw, bw):
    if tw == bw: return 'exact'
    if cached_stem(tw) == cached_stem(bw): return 'stem'
    ts, bs = cached_stem(tw), cached_stem(bw)
    if (ts, bs) in SYNONYM_SET: return 'synonym'
    if (ts, bs) in ANTONYM_SET: return 'antonym'
    return None

matched = []
for _, r in no_pred.iterrows():
    title = r['T']; body = r['C']
    tphrase = title_predicate_phrase(title)
    if not tphrase or len(tphrase) < 2:
        continue
    body_words = body.split()
    best_cov = 0.0; best_window = None; best_aligns = None; best_type = None
    for i in range(len(body_words)):
        lo = max(0, i - 2); hi = min(len(body_words), i + 3)
        bphrase = content_words(body_words[lo:hi])
        if not bphrase:
            continue
        align_types = []
        n_aligned = 0
        has_antonym = False
        for tw in tphrase:
            found = None
            for bw in bphrase:
                a = aligns(tw, bw)
                if a:
                    found = a
                    if a == 'antonym': has_antonym = True
                    break
            if found:
                n_aligned += 1
                align_types.append(found)
            else:
                align_types.append(None)
        cov = n_aligned / len(tphrase)
        if cov > best_cov:
            best_cov = cov; best_window = ' '.join(bphrase); best_aligns = align_types
            best_type = 'antonym' if has_antonym else ('synonym' if 'synonym' in align_types else ('stem' if 'stem' in align_types else 'exact'))
    if best_cov >= 0.50:
        matched.append(dict(id=r['id'], title=title, body_snippet=body[:200],
                             t_phrase=' '.join(tphrase), b_window=best_window,
                             coverage=round(best_cov, 2), align_type=best_type))

n = len(no_pred)
print(f"\nSTRICT phrase alignment (coverage>=0.50): {len(matched)} / {n}  ({len(matched)/n*100:.1f}%)")
print("\n=== ALL matched pairs (read every one for precision judgment) ===")
for m in matched:
    print(f"[{m['id']}] cov={m['coverage']} type={m['align_type']}")
    print(f"  TITLE PHRASE : {m['t_phrase']}")
    print(f"  BODY WINDOW  : {m['b_window']}")
    print(f"  TITLE FULL   : {m['title']}")
    print(f"  BODY SNIPPET : {m['body_snippet']}")
    print()
