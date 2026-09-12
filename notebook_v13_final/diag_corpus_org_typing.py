"""
Diagnostic: can the corpus itself tell us which ENT_POOL entities are ORGs, precisely?
For each entity already in ENT_POOL, count occurrences in train bodies that are preceded
(within 2 tokens) by a generic Indonesian institutional head-noun. Type as ORG only when
the cue is repeated AND forms a meaningful share of that entity's occurrences.
Prints coverage + a sample so precision can be judged BEFORE any CV run.
"""
import os, string as _s
from collections import Counter, defaultdict
import dill

d = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(d, '_cache.dill'), 'rb') as f:
    ns = dill.load(f)
train = ns['train']; ENT_POOL = ns['ENT_POOL']; entity_type = ns['entity_type']

ORG_CUES = {
    'pt','bank','universitas','yayasan','partai','kementerian','kemen','dinas','komisi','lembaga',
    'badan','kantor','polda','polres','polsek','polrestabes','kejaksaan','kejati','kejari',
    'pengadilan','rs','rsud','rsu','rumah','perusahaan','organisasi','asosiasi','ikatan',
    'persatuan','federasi','dewan','majelis','fraksi','satgas','tim','direktorat','ditjen',
    'kanwil','bpjs','otoritas','komite','forum','gerakan','aliansi','koalisi','serikat',
    'perkumpulan','himpunan','pusat','balai','sekretariat','markas','mabes','koramil','kodim',
    'korem','kodam','lanud','lanal','pangkalan','klinik','puskesmas','laboratorium','sekolah',
    'smp','sma','smk','sd','madrasah','pesantren','ponpes','masjid','gereja','pura','vihara',
    'hotel','mal','mall','pasar','stasiun','bandara','pelabuhan','terminal','maskapai',
}

occ = Counter(); cued = Counter(); cue_by_ent = defaultdict(Counter)
for body in train['C']:
    toks = [w.strip(_s.punctuation) for w in body.split()]
    for i, w in enumerate(toks):
        if w in ENT_POOL:
            occ[w] += 1
            prev = [toks[j].lower() for j in (i-1, i-2) if j >= 0]
            hit = [p for p in prev if p in ORG_CUES]
            if hit:
                cued[w] += 1; cue_by_ent[w][hit[0]] += 1

MIN_CUED, MIN_SHARE = 2, 0.30
new_org = {}
for w in ENT_POOL:
    if entity_type(w) != 'UNKNOWN': continue
    if cued[w] >= MIN_CUED and cued[w] / max(occ[w], 1) >= MIN_SHARE:
        new_org[w] = (cued[w], occ[w], cue_by_ent[w].most_common(2))

already = sum(1 for w in ENT_POOL if entity_type(w) != 'UNKNOWN')
print(f"ENT_POOL size            : {len(ENT_POOL)}")
print(f"already typed (gazetteer): {already}")
print(f"newly typed ORG (corpus) : {len(new_org)}   [cued>={MIN_CUED}, share>={MIN_SHARE}]")
print(f"typed coverage after     : {(already+len(new_org))/len(ENT_POOL):.1%}")

print("\n--- audit entities from the error decomposition ---")
for w in ['KPK','ICW','Polda','Kemenkes','Bareskrim','Satgas','Kemendikbud','Golkar','Nasdem','Kominfo']:
    print(f"  {w:12s} in_pool={w in ENT_POOL!s:5s} type_now={entity_type(w) if w in ENT_POOL else '-':8s} "
          f"cued={cued[w]}/{occ[w]}  -> {'ORG(new)' if w in new_org else ('already' if w in ENT_POOL and entity_type(w)!='UNKNOWN' else 'still UNKNOWN')}")

print("\n--- sample of newly typed ORG, sorted by occurrences (judge precision here) ---")
for w, (c, o, top) in sorted(new_org.items(), key=lambda kv: -kv[1][1])[:60]:
    print(f"  {w:22s} cued {c:3d}/{o:4d}  cues={top}")

print("\n--- lowest-share ones that still passed (most at risk of being wrong) ---")
for w, (c, o, top) in sorted(new_org.items(), key=lambda kv: kv[1][0]/kv[1][1])[:25]:
    print(f"  {w:22s} cued {c:3d}/{o:4d}  share={c/o:.2f}  cues={top}")
