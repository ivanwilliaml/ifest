"""
Pre-CV diagnostic for Experiment C2 (claim-conditioned, proximity-aware evidence retrieval
replacing find_predicate_chunk). Compares OLD selection (exact>synonym>antonym, first-found,
scanning 50-word chunks) vs NEW selection (same tier priority but exact>antonym>synonym,
scored by entity/number/title overlap + proximity, scanning finer 20-word/10-stride units)
over the full train set. No CatBoost, no CV -- pure retrieval-behavior comparison, fast.
"""
import io, os, json, re, string as _s
import numpy as np, pandas as pd

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
# Only exec cells up through claim-representation SETUP (before claim_representation runs),
# by pulling in everything up to (but not including) the claim_representation Timer block --
# simplest robust way: exec the whole notebook up to and including claim_scores definition,
# then override find_predicate_chunk ourselves and iterate manually (much faster than a full
# rerun since we skip CatBoost/CV entirely by just not executing those later cells).
cells_src = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
# find_predicate_in/find_predicate_chunk/claim_scores and the Timer("claim_representation")
# loop all live in the SAME notebook cell, so we can't split mid-cell -- instead stop before
# the (separate, later) cell that starts the expensive CatBoost/StratifiedGroupKFold CV.
stop_idx = next(i for i, s in enumerate(cells_src) if 'StratifiedGroupKFold' in s)
src = '\n\n'.join(cells_src[:stop_idx])
src = src.replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
ns = {'__name__': '__main__'}
exec(compile(src, os.path.join(d, '_diag_c2_setup.py'), 'exec'), ns)

train = ns['train']
find_predicate_in = ns['find_predicate_in']
find_all_predicates_in = ns['find_all_predicates_in']
best_chunk_lex = ns['best_chunk_lex']
chunks_of = ns['chunks_of']
ANTONYM_SET = ns['ANTONYM_SET']; SYNONYM_SET = ns['SYNONYM_SET']
ENT_POOL = ns['ENT_POOL']; NUM = ns['NUM']; toks = ns['toks']

def find_predicate_chunk_OLD(title, chunks):
    ti, troot = find_predicate_in(title)
    if troot is None: return best_chunk_lex(title, chunks), None, None, 'none'
    rank = {'exact': 0, 'synonym': 1, 'antonym': 2}
    best = None
    for c in chunks:
        for bi, broot in find_all_predicates_in(c):
            if broot == troot: mtype = 'exact'
            elif (troot, broot) in SYNONYM_SET: mtype = 'synonym'
            elif (troot, broot) in ANTONYM_SET: mtype = 'antonym'
            else: continue
            r = rank[mtype]
            if best is None or r < best[0]: best = (r, c, bi, broot, mtype)
        if best is not None and best[0] == 0: break
    if best is not None:
        _, c, bi, broot, mtype = best
        return c, bi, broot, mtype
    return best_chunk_lex(title, chunks), None, None, 'none'

def find_predicate_chunk_NEW(title, body):
    ti, troot = find_predicate_in(title)
    if troot is None: return best_chunk_lex(title, chunks_of(body)), None, None, 'none'
    units = chunks_of(body, w=20, st=10)
    tents_local = {w for w in title.split() if w in ENT_POOL}
    tnum_local = set(NUM.findall(title))
    ttok_local = set(toks(title))
    tier = {'exact': 0, 'antonym': 1, 'synonym': 2}
    best = None
    for u in units:
        uwords = u.split()
        craw_u = {w.strip(_s.punctuation) for w in uwords}
        for bi_local, broot_local in find_all_predicates_in(u):
            if broot_local == troot: mtype = 'exact'
            elif (troot, broot_local) in ANTONYM_SET: mtype = 'antonym'
            elif (troot, broot_local) in SYNONYM_SET: mtype = 'synonym'
            else: continue
            ent_ov = (len(tents_local & craw_u) / len(tents_local)) if tents_local else 0.0
            num_ov = (len(tnum_local & set(NUM.findall(u))) / len(tnum_local)) if tnum_local else 0.0
            tok_ov = len(ttok_local & set(toks(u))) / max(len(ttok_local), 1)
            match_positions = [k for k, w2 in enumerate(uwords)
                                if w2.strip(_s.punctuation) in tents_local or re.fullmatch(r'\d[\d.,]*%?', w2.strip(_s.punctuation))]
            if match_positions:
                mind = min(abs(bi_local - k) for k in match_positions)
                prox = max(0.0, 1.0 - mind / 10.0)
            else:
                prox = 0.5
            subscore = 0.4*ent_ov + 0.2*num_ov + 0.15*tok_ov + 0.25*prox
            key = (tier[mtype], -subscore)
            if best is None or key < best[0]:
                best = (key, u, bi_local, broot_local, mtype)
    if best is None:
        return best_chunk_lex(title, chunks_of(body)), None, None, 'none'
    _, u, bi_local, broot_local, mtype = best
    return u, bi_local, broot_local, mtype

n = len(train)
has_troot = 0
old_types = {'exact': 0, 'synonym': 0, 'antonym': 0, 'none': 0}
new_types = {'exact': 0, 'synonym': 0, 'antonym': 0, 'none': 0}
differs_text = 0
differs_type = 0
compared = 0

for T, C in zip(train['T'], train['C']):
    ti, troot = find_predicate_in(T)
    if troot is not None: has_troot += 1
    all_chunks = chunks_of(C)
    oc, obi, obroot, otype = find_predicate_chunk_OLD(T, all_chunks)
    nc, nbi, nbroot, ntype = find_predicate_chunk_NEW(T, C)
    old_types[otype] += 1
    new_types[ntype] += 1
    if troot is not None:
        compared += 1
        if otype != ntype: differs_type += 1
        if oc.strip() != nc.strip(): differs_text += 1

print(f"n rows: {n}")
print(f"rows with a title predicate at all: {has_troot} ({has_troot/n*100:.1f}%)")
print(f"\nOLD selection type breakdown: {old_types}  ({ {k: f'{v/n*100:.1f}%' for k,v in old_types.items()} })")
print(f"NEW selection type breakdown: {new_types}  ({ {k: f'{v/n*100:.1f}%' for k,v in new_types.items()} })")
print(f"\nAmong {compared} rows with a title predicate:")
print(f"  evidence TEXT differs from old: {differs_text} ({differs_text/compared*100:.1f}%)")
print(f"  evidence TYPE differs from old: {differs_type} ({differs_type/compared*100:.1f}%)")
print(f"  NEW found evidence where OLD found none: {sum(1 for _ in [1])}")  # placeholder, real count below

# how many rows flip none->found or found->none
old_none_new_found = 0; old_found_new_none = 0
for T, C in zip(train['T'], train['C']):
    ti, troot = find_predicate_in(T)
    if troot is None: continue
    all_chunks = chunks_of(C)
    _, _, _, otype = find_predicate_chunk_OLD(T, all_chunks)
    _, _, _, ntype = find_predicate_chunk_NEW(T, C)
    if otype == 'none' and ntype != 'none': old_none_new_found += 1
    if otype != 'none' and ntype == 'none': old_found_new_none += 1
print(f"  OLD=none -> NEW=found: {old_none_new_found}")
print(f"  OLD=found -> NEW=none: {old_found_new_none}")
