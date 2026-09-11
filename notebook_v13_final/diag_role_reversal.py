"""
Direction #3 diagnostic (entity-anchored role reversal): for rows where the title has
EXACTLY 2 known ENT_POOL entities and a matched predicate root, record which entity comes
first relative to the predicate. Scan the body for a window containing BOTH entities near
the SAME (or synonym/antonym) predicate root, and check if the entity order is swapped
relative to the predicate -- a structural proxy for "X did P to Y" vs "Y did P to X"
without a real dependency parser. Runs on ALL 288 current FP rows (not just the two
heuristic buckets) since role-reversal could hide inside any category. Coverage + manual
precision read BEFORE any CV code, per established discipline.
"""
import io, os, json, re, string as _s
import numpy as np, pandas as pd

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
src = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
src = src.replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
src = src.replace("/kaggle/working/submission.csv", os.path.join(d, 'local_out', 'submission.csv').replace("\\", "\\\\"))
ns = {'__name__': '__main__'}
exec(compile(src, os.path.join(d, '_dryrun_rr.py'), 'exec'), ns)

train = ns['train']; Ftr = ns['Ftr']; cold_mask = ns['cold_mask']
cold_oof = ns['cold_oof']; cold_y = ns['cold_y']; A_THR = ns['A_THR']
cold_pred = (cold_oof >= A_THR).astype(int)
fp_mask = (cold_y == 0) & (cold_pred == 1)
cold_idx = np.where(cold_mask)[0]
cold_train = train.iloc[cold_idx].reset_index(drop=True)
fp_train = cold_train[fp_mask].reset_index(drop=True)
print(f"Total FP (current best model): {len(fp_train)}")

find_predicate_in = ns['find_predicate_in']; find_all_predicates_in = ns['find_all_predicates_in']
ENT_POOL = ns['ENT_POOL']; SYNONYM_SET = ns['SYNONYM_SET']; ANTONYM_SET = ns['ANTONYM_SET']

def title_entities_and_predicate(title):
    words = title.split()
    ents = [(i, w.strip(_s.punctuation)) for i, w in enumerate(words) if w.strip(_s.punctuation) in ENT_POOL]
    if len(ents) != 2:
        return None
    ti, troot = find_predicate_in(title)
    if ti is None:
        return None
    (i1, e1), (i2, e2) = ents
    order = 'e1_first' if i1 < i2 else 'e2_first'
    pred_side = 'pred_after' if ti > max(i1, i2) else ('pred_before' if ti < min(i1, i2) else 'pred_between')
    return dict(e1=e1, e2=e2, troot=troot, order=order, pred_side=pred_side)

def body_order_for_pair(body, e1, e2, troot):
    words = body.split()
    best = None
    for i, w in enumerate(words):
        wl = w.strip(_s.punctuation).lower()
        if not wl:
            continue
        root = None
        for bi, broot in [(i, None)]:
            pass
        # check predicate match at this position via stemmed root membership
        from_stem = ns['cached_stem'](wl)
        mtype = None
        if from_stem == troot: mtype = 'exact'
        elif (troot, from_stem) in SYNONYM_SET: mtype = 'synonym'
        elif (troot, from_stem) in ANTONYM_SET: mtype = 'antonym'
        if mtype is None:
            continue
        lo, hi = max(0, i - 15), min(len(words), i + 15)
        window = words[lo:hi]
        pos_e1 = next((k for k, w2 in enumerate(window) if w2.strip(_s.punctuation) == e1), None)
        pos_e2 = next((k for k, w2 in enumerate(window) if w2.strip(_s.punctuation) == e2), None)
        if pos_e1 is None or pos_e2 is None:
            continue
        order = 'e1_first' if pos_e1 < pos_e2 else 'e2_first'
        if best is None or mtype == 'exact':
            best = dict(order=order, mtype=mtype, window=' '.join(window))
            if mtype == 'exact':
                break
    return best

candidates = []
for _, r in fp_train.iterrows():
    info = title_entities_and_predicate(r['T'])
    if info is None:
        continue
    bres = body_order_for_pair(r['C'], info['e1'], info['e2'], info['troot'])
    if bres is None:
        continue
    reversed_ = (bres['order'] != info['order'])
    candidates.append(dict(id=r['id'], title=r['T'], e1=info['e1'], e2=info['e2'],
                            title_order=info['order'], body_order=bres['order'],
                            match_type=bres['mtype'], reversed=reversed_, body_window=bres['window']))

n_total_scanned = len(fp_train)
n_with_2ent_pred = sum(1 for _, r in fp_train.iterrows() if title_entities_and_predicate(r['T']) is not None)
n_body_matched = len(candidates)
n_reversed = sum(1 for c in candidates if c['reversed'])
print(f"rows with exactly 2 title entities + matched predicate: {n_with_2ent_pred} / {n_total_scanned}")
print(f"of those, body window found with both entities near matching predicate: {n_body_matched}")
print(f"of those, order REVERSED (candidate role-swap signal): {n_reversed}")

print("\n=== ALL candidates with order comparison (read for precision) ===")
for c in candidates:
    flag = "*** REVERSED ***" if c['reversed'] else "same order"
    print(f"[{c['id']}] {flag}  match={c['match_type']}")
    print(f"  TITLE: {c['title']}")
    print(f"  entities: {c['e1']} / {c['e2']}  title_order={c['title_order']}  body_order={c['body_order']}")
    print(f"  BODY WINDOW: {c['body_window']}")
    print()
