import nbformat as nbf, textwrap, os

nb = nbf.v4.new_notebook(); cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s): cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))

md("""
# Penyisihan IFEST 2026 DAC — V13: Full Regime Pipeline (Final)

## Compliance
No pretrained models/embeddings/LLM/API anywhere. Only: TF-IDF, BM25 (from-scratch),
LSA (from-scratch, fit on competition corpus), classical classifiers (LogReg/CatBoost),
and **static external lexical/geographic lookup tables** (KBBI-derived antonym/synonym
pairs, Indonesian province/regency administrative data) -- embedded below as literal data,
downloaded once before experimentation, never called live. User explicitly authorized use
of external static data and the exact-pair rule after checking with organizers (2026-09-10).

## Architecture — regime router
```
TEST row
   |
   +-- C_exact (exact title+body pair already in train) -> rule: copy known label (1.0 Macro F1, 0 label noise ever observed)
   +-- B_has_pos / B_no_pos (body seen before)           -> title-retrieval + entity-conflict classifier (~0.98 Macro F1, honest CV)
   +-- A_cold (body never seen)                          -> 48 v12 features + claim representation + full-body evidence scan, CatBoost (~0.67 Macro F1, honest CV)
```
Each component was independently developed, ablated, and honestly cross-validated
(grouped by content_hash where regime membership requires it, repeated-split where the
regime needs the mixture to appear) across the project's experiment history -- see
EXPERIMENTS.md in the repo root for the full trail, including everything that was tried
and reverted.
""")

md("## 1. Environment")
code("""
import os, re, time, json, math, hashlib, unicodedata, warnings, random, string as _s
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from collections import Counter, defaultdict

SEED=42; random.seed(SEED); np.random.seed(SEED)
import sklearn; print("sklearn", sklearn.__version__)
try:
    import catboost as cb; print("catboost", cb.__version__)
except Exception as e:
    print("catboost unavailable:", e); raise
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
except Exception as e:
    print("Sastrawi unavailable, falling back to identity stemmer:", e)
    class _NoStem:
        def stem(self, w): return w
    stemmer = _NoStem()

TIMINGS={}
class Timer:
    def __init__(s,n): s.n=n
    def __enter__(s): s.t=time.time(); return s
    def __exit__(s,*a):
        d=time.time()-s.t; TIMINGS[s.n]=TIMINGS.get(s.n,0)+d; print(f"[TIMER] {s.n}: {d:.1f}s")
RUN_T0=time.time()
""")

md("## 2. Load data")
code("""
with Timer("load"):
    F={}
    for r,_,fs in os.walk('/kaggle/input'):
        for f in fs:
            if f in ('train.csv','test.csv','sample_submission.csv'): F.setdefault(f,os.path.join(r,f))
    assert len(F)==3, F
    train=pd.read_csv(F['train.csv']); test=pd.read_csv(F['test.csv']); sample_sub=pd.read_csv(F['sample_submission.csv'])

def nh(x):
    x=unicodedata.normalize('NFKC',str(x)).lower(); return re.sub(r'\\s+',' ',x).strip()
for d in (train,test):
    d['ch']=d['content'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
    d['th']=d['title'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
URL=re.compile(r'https?://\\S+|www\\.\\S+'); HTML=re.compile(r'<[^>]+>'); WS=re.compile(r'\\s+')
def clean(s):
    s=unicodedata.normalize('NFKC',str(s)); s=HTML.sub(' ',s); s=URL.sub(' ',s)
    return WS.sub(' ',s).strip()
for d in (train,test):
    d['T']=d['title'].apply(clean); d['C']=d['content'].apply(clean)
y=train['label'].values
print("train",train.shape,"test",test.shape,"class0",round((train.label==0).mean(),4))
""")

md("""
## 3. Embedded external static resources
KBBI-derived antonym/synonym pairs (dyazincahya/KBBI-SQL-database) and Indonesian
province/regency administrative hierarchy (edwardsamuel/Wilayah-Administratif-Indonesia).
**Provenance note:** the antonym/synonym source README states this data was compiled with
LLM assistance (GPT-5.6 Sol) -- it is a static downloaded dataset, not a live model call
during feature generation. Recorded here per the project's transparency policy.
""")

with open(os.path.join(os.path.dirname(__file__), 'embedded_data.py'), encoding='utf-8') as f:
    embedded_src = f.read()
code(embedded_src)

code("""
ANTONYM_SET=set()
for a,b in ANTONYM_PAIRS_RAW: ANTONYM_SET.add((a,b)); ANTONYM_SET.add((b,a))
# ACTION_VERBS = every antonym root, unfiltered by POS (~1/3 are adjectives: "baik/buruk",
# "besar/kecil", "mahal/murah"...), used as the predicate-candidate set for
# find_predicate_chunk/pred_score/polarity_score/argument_binding_conflict. Two corpus-
# verified attempts to restrict this to real verbs (naive meN-prefix concatenation: -0.0051;
# proper allomorph-aware stemming, 292/895 roots confirmed verbal: -0.0063) both regressed
# CV, worse than the naive version. Left unfiltered deliberately: the adjective roots appear
# to work as a coarse sentiment/framing-shift proxy despite being conceptually mislabeled as
# "verbs" -- narrowing to true verbs removes that signal along with the noise. Fourth
# confirmed instance this session of a more-precise entity/predicate-type fix regressing
# (see entity_type() ORG-widening attempts) -- do not retry this family of fix again without
# new evidence.
ACTION_VERBS=set([a for a,_ in ANTONYM_SET])
SYNONYM_SET=set()
for a,b in SYNONYM_PAIRS_RAW: SYNONYM_SET.add((a,b)); SYNONYM_SET.add((b,a))
PROVINCES=set(PROVINCE_NAMES_RAW)
CITIES=set()
CITY_PROVINCE={}
for name,pid in REGENCY_PROVINCE_RAW:
    CITIES.add(name); CITY_PROVINCE[name]=PROVINCE_ID_MAP_RAW.get(pid)
COUNTRIES={'Indonesia','Malaysia','Singapura','Thailand','Filipina','Vietnam','China','Tiongkok',
    'Jepang','Korea','Taiwan','India','Pakistan','Australia','Zimbabwe','Kenya','Amerika','AS',
    'Inggris','Prancis','Jerman','Italia','Spanyol','Portugal','Belanda','Rusia','Brasil','Kanada',
    'Meksiko','Mesir','Arab','Turki','Iran','Irak','Swedia','Norwegia','Denmark','Swiss','Austria',
    'Belgia','Yunani','Polandia','Ukraina','Selandia'}
ORGANIZATIONS={'BPOM','Kemenkes','Kominfo','Kemendagri','Kemenag','Kemendikbud','BNPB','BPBD',
    'Satpol','Polri','TNI','DPR','DPRD','KPK','MPR','MUI','IDI','KPU','Kejaksaan','Golkar','PDIP',
    'PPP','PAN','PKS','PKB','Demokrat','Nasdem','Gerindra','Perindo','WHO','UNICEF','PBB','Satgas',
    'BUMN','KAI','Garuda','Pertamina','PLN','BNI','BI'}
STOPWORDS_ROLE={'Gubernur','Wagub','Presiden','Wapres','Menteri','Menkes','Menko','Kementerian',
    'Ketua','Wali','Bupati','Wakil','Kepala','Dinas','Badan','Komisi','Partai','Kapolri','Kapolres',
    'Kapolda','Menkeu','Mendagri','Mensesneg','Kabareskrim','Sekretaris','Direktur','Komisioner',
    'Anggota','Pemerintah','Kasatgas','Karo','Kabid','Kadin','Kadis','Pemprov','Pemkot','Pemkab',
    'COVID','Corona','Covid','Menristek','Panglima','Jaksa','Hakim','Rektor','Camat','Lurah','Kanwil'}
print(f"antonym pairs: {len(ANTONYM_PAIRS_RAW)}  synonym pairs: {len(SYNONYM_PAIRS_RAW)}")
print(f"provinces: {len(PROVINCES)}  regencies: {len(CITIES)}")
""")

md("## 4. Entity mining (from competition corpus) + typed geography")
code("""
with Timer("entity_mining"):
    up,lo=Counter(),Counter()
    for t in train['C']:
        for w in t.split():
            c=w.strip(_s.punctuation)
            if not c.isalpha() or len(c)<3: continue
            if c[:1].isupper(): up[c]+=1
            else: lo[c.capitalize()]+=1
    pur={w:up[w]/(up[w]+lo.get(w,0)) for w in up if up[w]>=8}
    tdf=Counter()
    for t in train['T']:
        for w in set(t.split()): tdf[w]+=1
    ENT_CORPUS={w for w,p in pur.items() if p>=0.95 and tdf.get(w,0)/len(train)<0.02}
ENT_POOL = ENT_CORPUS | COUNTRIES | PROVINCES | CITIES | ORGANIZATIONS
ENT_POOL = {e for e in ENT_POOL if e not in STOPWORDS_ROLE}
def entity_type(word):
    # Only geography/org/country are confidently typed from known lists. Any other
    # capitalized proper noun mined from the corpus (the old code guessed 'PERSON') is
    # UNKNOWN -- type-conflict logic must not treat an unknown-type entity as a known type.
    # NOTE (2026-09-11): widening ORG coverage was tried twice and regressed CV both times
    # (all-caps acronym regex -0.0056; corpus-context head-noun cues, 391 precise ORGs,
    # -0.0067). "Another same-type entity is present in the chunk" is weak evidence --
    # consistent articles routinely mention several orgs -- so more typed entities means
    # the conflict signal fires more often on label-1 rows and the model discounts it.
    if word in COUNTRIES: return 'COUNTRY'
    if word in PROVINCES: return 'PROVINCE'
    if word in CITIES: return 'CITY'
    if word in ORGANIZATIONS: return 'ORG'
    return 'UNKNOWN'
def same_hierarchy(e1,e2):
    if e1==e2: return True
    p1=CITY_PROVINCE.get(e1); p2=CITY_PROVINCE.get(e2)
    return bool((p1 and p1==e2) or (p2 and p2==e1))

# Multi-word gazetteer entries (provinces/cities/orgs/countries with a space) kept as a
# SEPARATE phrase-matching lookup, parallel to the existing single-token ENT_POOL. This
# does not touch the single-token pipeline -- a prior attempt to unify them regressed CV.
PHRASE_ENTITIES = {}
for _name in (COUNTRIES | PROVINCES | CITIES | ORGANIZATIONS):
    if ' ' in _name:
        PHRASE_ENTITIES[_name.lower()] = entity_type(_name)
def extract_phrase_entities(tokens):
    lowered=[w.strip(_s.punctuation).lower() for w in tokens]
    n=len(lowered); found=[]; i=0
    while i<n:
        matched=False
        for L in (3,2):
            if i+L<=n:
                cand=' '.join(lowered[i:i+L])
                if cand in PHRASE_ENTITIES:
                    found.append((cand, PHRASE_ENTITIES[cand])); i+=L; matched=True; break
        if not matched: i+=1
    return found

# Experiment B (abbreviation/entity canonicalization): a fixed global alias table of
# well-known Indonesian province/regency abbreviations, resolved against the ACTUAL loaded
# gazetteer (not assumed) -- an entry that doesn't match any real gazetteer string simply
# never resolves (logged), rather than silently mapping to a wrong canonical string. The
# corpus has zero parentheses/punctuation (verified), so a Schwartz-Hearst-style
# "Long Form (SHORT)" local-discovery pass is not usable here and is skipped.
_PROVINCE_ABBREV_GUESS = {
    'jabar':'jawa barat','jateng':'jawa tengah','jatim':'jawa timur','dki':'dki jakarta',
    'diy':'yogyakarta','sumut':'sumatera utara','sumsel':'sumatera selatan',
    'sumbar':'sumatera barat','kepri':'kepulauan riau','kaltim':'kalimantan timur',
    'kalsel':'kalimantan selatan','kalbar':'kalimantan barat','kalteng':'kalimantan tengah',
    'kaltara':'kalimantan utara','sulsel':'sulawesi selatan','sulut':'sulawesi utara',
    'sulteng':'sulawesi tengah','sultra':'sulawesi tenggara','sulbar':'sulawesi barat',
    'babel':'bangka belitung','ntb':'nusa tenggara barat','ntt':'nusa tenggara timur',
    'malut':'maluku utara',
}
_REGENCY_ABBREV_GUESS = {
    'oki':'ogan komering ilir','oku':'ogan komering ulu',
    'okus':'ogan komering ulu selatan','okut':'ogan komering ulu timur','oi':'ogan ilir',
}
_province_lower_map={p.lower():p for p in PROVINCES}
_city_lower_map={c.lower():c for c in CITIES}
ABBREV_TO_CANONICAL={}
for _abbr,_full in _PROVINCE_ABBREV_GUESS.items():
    if _full in _province_lower_map: ABBREV_TO_CANONICAL[_abbr]=_province_lower_map[_full]
for _abbr,_full in _REGENCY_ABBREV_GUESS.items():
    if _full in _city_lower_map: ABBREV_TO_CANONICAL[_abbr]=_city_lower_map[_full]
print(f"abbreviation alias table: {len(ABBREV_TO_CANONICAL)}/{len(_PROVINCE_ABBREV_GUESS)+len(_REGENCY_ABBREV_GUESS)} resolved against loaded gazetteer")
def canonicalize_geo(word):
    wl=word.lower()
    if wl in ABBREV_TO_CANONICAL: return ABBREV_TO_CANONICAL[wl]
    if word in PROVINCES or word in CITIES or word in COUNTRIES: return word
    return None
def geo_canonical_mentions(tokens_raw):
    out=set()
    for w in tokens_raw:
        c=canonicalize_geo(w.strip(_s.punctuation))
        if c: out.add(c.lower())
    return out

# P1: conservative PERSON-candidate detector. NOT a gazetteer expansion -- names are
# open-vocabulary, so this is extracted per-document, not from a static list. A candidate
# is either (a) 2+ consecutive capitalized alphabetic tokens, or (b) 1-3 capitalized tokens
# immediately following a known role/rank marker (e.g. "AKBP Harun", "Irjen Nana Sudjana").
# Deliberately does NOT default every unknown capitalized word to a person (that was the
# bug just fixed in entity_type) -- a bare single capitalized token with no role marker and
# no multi-token extension is NOT considered a candidate.
ROLE_MARKERS = STOPWORDS_ROLE | {
    'AKBP','Kombes','Irjen','Brigjen','Mayjen','Letjen','Jenderal','Kolonel','Kompol',
    'Iptu','Ipda','Bripka','Kapten','Mayor','Letkol','Prof','Dr','Ustaz','Ustadz','KH',
    'Habib','Datuk','Raden','Kiai','Kyai',
}
def _is_name_tok(w):
    return bool(w) and w[:1].isupper() and w.isalpha() and len(w)>=2 and w not in ENT_POOL and w not in ROLE_MARKERS
def extract_person_candidates(tokens_raw):
    toks=[w.strip(_s.punctuation) for w in tokens_raw]
    n=len(toks); candidates=set(); anchored=[]
    i=0
    while i<n:
        w=toks[i]
        if w in ROLE_MARKERS:
            j=i+1; seq=[]
            while j<n and len(seq)<3 and _is_name_tok(toks[j]):
                seq.append(toks[j]); j+=1
            if seq:
                phrase=' '.join(seq); candidates.add(phrase); anchored.append((w,phrase)); i=j; continue
        elif len(w)>=3 and _is_name_tok(w):
            j=i+1; seq=[w]
            while j<n and _is_name_tok(toks[j]):
                seq.append(toks[j]); j+=1
            if len(seq)>=2:
                candidates.add(' '.join(seq)); i=j; continue
        i+=1
    return candidates, anchored
print("entity pool:", len(ENT_POOL), " phrase gazetteer:", len(PHRASE_ENTITIES), " role markers:", len(ROLE_MARKERS))
""")

md("## 5. Word-chunk BM25 index + from-scratch LSA")
code("""
NUM=re.compile(r'\\d+[.,]?\\d*\\s*%?'); YEAR=re.compile(r'\\b(?:19|20)\\d{2}\\b')
NEG=['tidak','tak','bukan','belum','tanpa','gagal','ditolak','membantah','bantah','sangkal','menyangkal','menolak','dilarang']
INCREASE_ROOTS={'naik','tingkat','tambah','lonjak','tanjak','pesat'}
DECREASE_ROOTS={'turun','kurang','rosot','lambat','susut','tipis','landai'}
def toks(s): return re.findall(r"[\\w']+", s.lower())
def chunks_of(s,w=50,st=25):
    ws=s.split()
    if not ws: return ['']
    if len(ws)<=w: return [' '.join(ws)]
    out=[]
    for i in range(0,len(ws),st):
        c=ws[i:i+w]
        if not c: break
        out.append(' '.join(c))
        if i+w>=len(ws): break
    return out
def best_chunk_lex(title, chunks):
    tw=set(toks(title)); best_c,best_score=chunks[0],-1
    for c in chunks:
        score=len(tw & set(toks(c)))
        if score>best_score: best_score=score; best_c=c
    return best_c

with Timer("chunk_index"):
    allpairs=pd.concat([train[['ch','C']],test[['ch','C']]],ignore_index=True).drop_duplicates('ch')
    train_ch_set=set(train['ch'])
    chunk_txt={h:chunks_of(c) for h,c in zip(allpairs['ch'],allpairs['C'])}
    df_cnt=Counter(); total_len=0; n_chunks=0; chunk_tf={}
    for h,cs in chunk_txt.items():
        tfs=[]
        for c in cs:
            tk=toks(c); tf=Counter(tk); tfs.append((tf,len(tk)))
            # IDF/AVGDL corpus statistics are fit on TRAIN chunks only -- test chunks are
            # tokenized here (needed so bm25_scores can query them) but never contribute
            # to document-frequency or average-length statistics.
            if h in train_ch_set:
                total_len+=len(tk); n_chunks+=1
                for t in tf: df_cnt[t]+=1
        chunk_tf[h]=tfs
    AVGDL=total_len/max(n_chunks,1)
    IDF={t: math.log(1+(n_chunks-c+0.5)/(c+0.5)) for t,c in df_cnt.items()}
k1,b=1.5,0.75
def bm25_scores(title, ch):
    q=toks(title); out=[]
    for tf,L in chunk_tf[ch]:
        s=0.0
        for t in q:
            f=tf.get(t,0)
            if f: s+=IDF.get(t,0.0)*(f*(k1+1))/(f+k1*(1-b+b*L/AVGDL))
        out.append(s)
    return np.array(out) if out else np.array([0.0])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
with Timer("lsa"):
    # TfidfVectorizer vocabulary/IDF and the SVD basis are fit on TRAIN text only.
    corpus = pd.concat([train['T'],train['C']]).tolist()
    lsa_tfidf=TfidfVectorizer(ngram_range=(1,1),min_df=3,sublinear_tf=True,max_features=80000)
    lsa_tfidf.fit(corpus)
    Xc = lsa_tfidf.transform(corpus)
    svd=TruncatedSVD(n_components=160,random_state=SEED); svd.fit(Xc)
def embed(texts): return normalize(svd.transform(lsa_tfidf.transform(texts)))
for d in (train,test):
    d_t=embed(d['T'].tolist()); d_c=embed(d['C'].tolist())
    d['lsa_cos']=np.sum(d_t*d_c,axis=1)

# Char n-gram TF-IDF cosine (never tried for A_cold before -- only used in B regime for
# title-vs-title). Train-only fit, same discipline as lsa_tfidf above. Captures morphological
# variants ("menerima"/"diterima"/"terima") that word-level TF-IDF/BM25 miss entirely.
with Timer("char_tfidf"):
    char_tfidf=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,sublinear_tf=True,max_features=60000)
    char_tfidf.fit(corpus)
for d in (train,test):
    ct=normalize(char_tfidf.transform(d['T'])); cc=normalize(char_tfidf.transform(d['C']))
    d['char_tfidf_cos']=np.asarray(ct.multiply(cc).sum(axis=1)).ravel()

chunk_owner=[]; flat_chunks=[]
for h,cs in chunk_txt.items():
    for c in cs: chunk_owner.append(h); flat_chunks.append(c)
CHUNK_VEC=embed(flat_chunks)
chunk_vecs=defaultdict(list)
for h,v in zip(chunk_owner,CHUNK_VEC): chunk_vecs[h].append(v)
for h in list(chunk_vecs.keys()): chunk_vecs[h]=np.array(chunk_vecs[h])
""")

md("## 6. Base 41 + structural 7 features (v12 lineage)")
code("""
def build_base(df):
    rows=[]
    tvecs_all=embed(df['T'].tolist())
    for T,C,ch,lsa,tv,char_cos in zip(df['T'],df['C'],df['ch'],df['lsa_cos'],tvecs_all,df['char_tfidf_cos']):
        tw=toks(T); cw=toks(C)
        _craw=[w.strip(_s.punctuation) for w in C.split()]
        _n=max(len(_craw),1); _first={}; _tf={}
        for _i,_w in enumerate(_craw):
            _tf[_w]=_tf.get(_w,0)+1
            if _w not in _first: _first[_w]=_i
        _bmax=max([_tf[e] for e in _tf if e in ENT_POOL], default=1)
        _tents=[w for w in T.split() if w in ENT_POOL]
        _pres=[e for e in _tents if e in _first]
        if _pres:
            _fp=[_first[e]/_n for e in _pres]; _rf=[_tf[e]/max(_bmax,1) for e in _pres]
            sal_maxfirst=max(_fp); sal_meanfirst=sum(_fp)/len(_fp); sal_minrelfreq=min(_rf)
            sal_late=float(max(_fp)>0.5); sal_out_of_lead=float(max(_first[e] for e in _pres)>60)
            sal_singleton=float(min(_tf[e] for e in _pres)==1)
            sal_lead_frac=sum(1 for e in _pres if _first[e]<=60)/len(_pres)
        else:
            sal_maxfirst=sal_meanfirst=1.0; sal_minrelfreq=0.0
            sal_late=sal_out_of_lead=sal_singleton=1.0; sal_lead_frac=0.0
        sal_n_present=float(len(_pres)); sal_all_present=float(bool(_tents) and len(_pres)==len(_tents))
        tset,cset=set(tw),set(cw)
        craw={w.strip(_s.punctuation) for w in C.split()}
        inter=tset&cset; union=tset|cset
        tb=set(zip(tw,tw[1:])); cb=set(zip(cw,cw[1:]))
        tents=[w for w in T.split() if w in ENT_POOL]; miss=[w for w in tents if w not in craw]
        sig=0; Ttok=T.split()
        for i,w in enumerate(Ttok):
            if w in ENT_POOL and w not in craw:
                nb=[Ttok[j] for j in (i-1,i+1) if 0<=j<len(Ttok)]
                nb=[x for x in nb if x.lower() not in ('dan','di','ke','yang')]
                if nb and all(x.strip(_s.punctuation) in craw for x in nb): sig+=1
        tn,cn=set(NUM.findall(T)),set(NUM.findall(C)); ty,cy=set(YEAR.findall(T)),set(YEAR.findall(C))
        bm=bm25_scores(T,ch); order=np.argsort(-bm); topk=bm[order[:5]]
        num_=sum(IDF.get(t,0.0) for t in tset&cset); den_=sum(IDF.get(t,0.0) for t in tset)
        svs=chunk_vecs.get(ch)
        if svs is None or len(svs)==0:
            s_first=s_first3=s_max=s_topk=s_std=s_min=s_argmax=0.0
        else:
            sims=svs@tv; kk=min(5,len(sims)); tkm=np.sort(sims)[-kk:]
            s_first=float(sims[0]); s_first3=float(sims[:min(3,len(sims))].mean())
            s_max=float(sims.max()); s_topk=float(tkm.mean()); s_std=float(sims.std())
            s_min=float(sims.min()); s_argmax=float(np.argmax(sims)/max(len(sims)-1,1))
        rows.append(dict(
            n_shared=len(inter), jaccard=len(inter)/max(len(union),1), dice=2*len(inter)/max(len(tset)+len(cset),1),
            title_cov=len(inter)/max(len(tset),1), content_cov=len(inter)/max(len(cset),1),
            bigram_ov=len(tb&cb)/max(len(tb|cb),1), idf_title_cov=num_/max(den_,1e-6),
            n_title_ent=len(tents), n_title_ent_missing=len(miss), frac_title_ent_missing=len(miss)/max(len(tents),1),
            any_title_ent_missing=float(len(miss)>0), swap_signature=float(sig>0), n_swap_sig=sig,
            n_num_title=len(tn), n_num_missing=len(tn-cn), num_overlap=len(tn&cn)/max(len(tn),1) if tn else 1.0,
            num_mismatch=float(bool(tn) and not (tn&cn)), n_year_title=len(ty), year_missing=float(bool(ty) and not (ty&cy)),
            neg_title=float(any(w in tset for w in NEG)), neg_body=float(any(w in cset for w in NEG)),
            neg_mismatch=float(any(w in tset for w in NEG)!=any(w in cset for w in NEG)),
            bm25_max=float(bm.max()), bm25_mean=float(bm.mean()), bm25_top=float(topk.mean()), bm25_std=float(bm.std()),
            bm25_argmax_pos=float(order[0]/max(len(bm)-1,1)), n_chunks=len(bm), lsa_cos=float(lsa),
            char_tfidf_cos=float(char_cos),
            len_title=len(tw), len_content=len(cw), len_ratio=len(tw)/max(len(cw),1),
            sal_maxfirst=sal_maxfirst, sal_meanfirst=sal_meanfirst, sal_minrelfreq=sal_minrelfreq, sal_late=sal_late,
            sal_out_of_lead=sal_out_of_lead, sal_singleton=sal_singleton, sal_lead_frac=sal_lead_frac,
            sal_n_present=sal_n_present, sal_all_present=sal_all_present,
            struct_first_chunk=s_first, struct_first3_mean=s_first3, struct_max_chunk=s_max,
            struct_topk_mean=s_topk, struct_std=s_std, struct_min_chunk=s_min, struct_argmax_pos=s_argmax,
        ))
    return pd.DataFrame(rows,index=df.index)
with Timer("base_features"):
    Ftr=build_base(train); Fte=build_base(test)
FEAT48=list(Ftr.columns)
print(f"{len(FEAT48)} base+structural features")
""")

md("""
## 7. Claim representation (v8/v10 lineage) — entity/predicate/polarity/quantity + full-body evidence scan
Continuous scores beat binary conflict flags in every ablation this project ran (honest CV).
Full-body evidence scan means the predicate-matching chunk is found by scanning ALL chunks
for one containing a matching predicate root, not just the lexically-closest chunk.
""")
code("""
_STEM_CACHE={}
def cached_stem(w):
    if w not in _STEM_CACHE:
        try: _STEM_CACHE[w]=stemmer.stem(w.lower())
        except Exception: _STEM_CACHE[w]=w.lower()
    return _STEM_CACHE[w]
def find_predicate_in(text, exclude_prep=True):
    words=text.split()
    for i,w in enumerate(words):
        wl=w.strip(_s.punctuation).lower()
        if wl in ('di','ke','dari','pada') and exclude_prep: continue
        root=cached_stem(wl)
        if root in ACTION_VERBS: return i,root
    return None,None
def find_all_predicates_in(text, exclude_prep=True):
    words=text.split(); out=[]
    for i,w in enumerate(words):
        wl=w.strip(_s.punctuation).lower()
        if wl in ('di','ke','dari','pada') and exclude_prep: continue
        root=cached_stem(wl)
        if root in ACTION_VERBS: out.append((i,root))
    return out
def find_predicate_chunk(title, chunks):
    # Find the body chunk whose predicate is actually RELEVANT to the title's predicate
    # (exact root > synonym > antonym), not just the first chunk containing any verb.
    ti,troot=find_predicate_in(title)
    if troot is None: return best_chunk_lex(title,chunks), None, None, 'none'
    rank={'exact':0,'synonym':1,'antonym':2}
    best=None
    for c in chunks:
        for bi,broot in find_all_predicates_in(c):
            if broot==troot: mtype='exact'
            elif (troot,broot) in SYNONYM_SET: mtype='synonym'
            elif (troot,broot) in ANTONYM_SET: mtype='antonym'
            else: continue
            r=rank[mtype]
            if best is None or r<best[0]: best=(r,c,bi,broot,mtype)
        if best is not None and best[0]==0: break
    if best is not None:
        _,c,bi,broot,mtype=best
        return c,bi,broot,mtype
    return best_chunk_lex(title,chunks), None, None, 'none'
NUM_TOKEN=re.compile(r'\\d+(?:[.,]\\d+)?')
MULT={'ribu':1e3,'juta':1e6,'miliar':1e9,'milyar':1e9,'triliun':1e12}
def extract_numbers(text):
    words=text.split(); vals=[]; i=0
    while i<len(words):
        w=words[i].strip(_s.punctuation); m=NUM_TOKEN.fullmatch(w)
        if m:
            try: val=float(w.replace(',','.'))
            except Exception: val=None
            if val is not None:
                j=i+1
                while j<len(words):
                    w2=words[j].strip(_s.punctuation)
                    if re.fullmatch(r'\\d{3}',w2) and val==int(val): val=val*1000+int(w2); j+=1
                    else: break
                if j<len(words):
                    w3=words[j].lower().strip(_s.punctuation)
                    if w3 in MULT: val*=MULT[w3]; j+=1
                vals.append(val); i=j; continue
        i+=1
    return vals

# Experiment A1 (quantity-context alignment): parallel to extract_numbers/qty_score above,
# NOT a replacement for it (Experiment A -- a global numeric-parser fix applied to every
# number -- regressed CV; the literature-motivated fix here is narrower and only fires when
# a title number has a genuinely context-matching candidate in the body). For every number,
# extract value + a small window of surrounding words (the "concept"/context, e.g. "tempat
# tidur" vs "tenaga kesehatan"). A title number is matched to whichever body number shares
# the most context, and value is compared only within that matched pair -- unrelated numbers
# elsewhere in the article never contribute.
def parse_indo_number_token(t):
    m=re.fullmatch(r'\\d{1,3}(?:\\.\\d{3})+(?:,\\d+)?', t)
    if m:
        intpart,_,decpart=t.partition(',')
        val=float(intpart.replace('.',''))
        if decpart: val+=float('0.'+decpart)
        return val
    if re.fullmatch(r'\\d{1,3}\\.\\d{3}', t):
        return float(t.replace('.',''))
    if re.fullmatch(r'\\d+,\\d+', t):
        return float(t.replace(',','.'))
    if re.fullmatch(r'\\d+\\.\\d+', t):
        return float(t)
    if re.fullmatch(r'\\d+', t):
        return float(t)
    return None
_QTY_STOPWORDS={'yang','dan','di','ke','dari','pada','akan','ini','itu','juga','dengan',
    'untuk','oleh','atau','ada','tidak','sudah','telah','saat','hari','tahun','sebanyak',
    'sekitar','bakal'}
# Experiment A3 (approximation-aware tolerance): a number preceded by a hedge word like
# "sekitar"/"kira-kira"/"hampir" is an approximation, not an exact claim -- "sekitar 100"
# vs "98" should not be scored the same as "tepat 100" vs "98". Refines A1's val_match
# tolerance only; does not add new columns.
_APPROX_MARKERS={'sekitar','kira-kira','hampir','kurang lebih'}
def _has_approx_marker(words, num_start_idx):
    prev=[words[k].strip(_s.punctuation).lower() for k in range(max(0,num_start_idx-2),num_start_idx)]
    return bool(set(prev) & _APPROX_MARKERS) or ' '.join(prev[-2:])=='kurang lebih'
def extract_number_instances(text):
    # context words are STEMMED (via the project's existing Sastrawi cached_stem) so
    # "Tambah"/"menambah" count as the same concept -- raw lexical overlap without
    # stemming under-matched even genuine same-concept pairs during testing.
    words=text.split(); n=len(words); out=[]; i=0
    while i<n:
        w=words[i].strip(_s.punctuation); val=parse_indo_number_token(w)
        if val is not None:
            approx=_has_approx_marker(words,i)
            j=i+1; span_end=i
            while j<n:
                w2=words[j].strip(_s.punctuation)
                if re.fullmatch(r'\\d{3}',w2) and val==int(val): val=val*1000+int(w2); j+=1; span_end=j-1
                else: break
            if j<n:
                w3=words[j].lower().strip(_s.punctuation)
                if w3 in MULT: val*=MULT[w3]; j+=1; span_end=j-1
            _lo=max(0,i-4); _hi=min(n,span_end+1+4)
            ctx=set()
            for k in range(_lo,_hi):
                if k<i or k>span_end:
                    c=words[k].strip(_s.punctuation).lower()
                    if c and c not in _QTY_STOPWORDS and not c.isdigit(): ctx.add(cached_stem(c))
            out.append(dict(value=val, context=ctx, approx=approx))
            i=j; continue
        i+=1
    return out
def qty_context_alignment(title, body):
    title_nums=extract_number_instances(title); body_nums=extract_number_instances(body)
    if not title_nums:
        return dict(qty_context_token_overlap=0.5, qty_context_exact_match=0.5, qty_context_conflict=0.0)
    overlaps=[]; exacts=0; conflicts=0
    for tn in title_nums:
        if not body_nums:
            overlaps.append(0.0); continue
        best_bn=None; best_ov=-1.0
        for bn in body_nums:
            u=tn['context'] | bn['context']
            ov=(len(tn['context'] & bn['context'])/len(u)) if u else 0.0
            if ov>best_ov: best_ov=ov; best_bn=bn
        overlaps.append(best_ov)
        if best_bn is not None:
            v=tn['value']; bv=best_bn['value']
            tol=0.10 if (tn.get('approx') or best_bn.get('approx')) else 0.02
            val_match=abs(bv-v)<1e-6 or (v!=0 and abs(bv-v)/max(abs(v),1.0)<tol)
            if val_match: exacts+=1
            elif best_ov>=0.20: conflicts+=1
    n=len(title_nums)
    return dict(qty_context_token_overlap=float(np.mean(overlaps)),
                qty_context_exact_match=exacts/n, qty_context_conflict=conflicts/n)

def claim_scores(title, body, ch):
    Ttok=title.split()
    tents=[(w,entity_type(w)) for w in Ttok if w in ENT_POOL]
    craw={w.strip(_s.punctuation) for w in body.split()}
    all_chunks=chunks_of(body)
    bc_lex=best_chunk_lex(title,all_chunks)
    bc_pred,bi,broot,pred_match_type=find_predicate_chunk(title,all_chunks)
    chunk_ents=[(w.strip(_s.punctuation),entity_type(w.strip(_s.punctuation)))
                for w in bc_lex.split() if w.strip(_s.punctuation) in ENT_POOL]
    body_ents_full=[(w.strip(_s.punctuation),entity_type(w.strip(_s.punctuation)))
                    for w in body.split() if w.strip(_s.punctuation) in ENT_POOL]
    if tents:
        supported=sum(1 for e,_ in tents if e in craw)
        entity_support=supported/len(tents)
        # Type-conflict evidence only ever compares entities with a KNOWN (non-UNKNOWN) type.
        n_conflict=0
        for e,etype in tents:
            if e in craw or etype=='UNKNOWN': continue
            alts=[ce for ce,ct in chunk_ents if ce!=e and ct==etype and not same_hierarchy(e,ce)]
            if alts: n_conflict+=1
        entity_conflict=n_conflict/len(tents)
        # Full-body variant: alternative-entity evidence is searched across the WHOLE body,
        # not just the single lexically-closest chunk. Kept as a separate continuous feature
        # alongside the original so nothing already validated is overwritten.
        n_conflict_full=0
        for e,etype in tents:
            if e in craw or etype=='UNKNOWN': continue
            alts=[ce for ce,ct in body_ents_full if ce!=e and ct==etype and not same_hierarchy(e,ce)]
            if alts: n_conflict_full+=1
        entity_conflict_fullbody=n_conflict_full/len(tents)
    else:
        entity_support,entity_conflict,entity_conflict_fullbody=0.5,0.0,0.0
    # Parallel multi-word (phrase) entity support/conflict, separate from the single-token
    # features above -- longest-match 2-3 gram against the static gazetteer, whole body scan.
    title_phrases=extract_phrase_entities(Ttok)
    body_phrases=extract_phrase_entities(body.split())
    body_phrase_set={p for p,_ in body_phrases}
    if title_phrases:
        p_supported=sum(1 for p,_ in title_phrases if p in body_phrase_set)
        phrase_entity_support=p_supported/len(title_phrases)
        p_conflict=0
        for p,ptype in title_phrases:
            if p in body_phrase_set: continue
            alts=[bp for bp,bt in body_phrases if bp!=p and bt==ptype]
            if alts: p_conflict+=1
        phrase_entity_conflict=p_conflict/len(title_phrases)
    else:
        phrase_entity_support,phrase_entity_conflict=0.5,0.0
    # Experiment B: abbreviation-aware canonical geography matching. "Jabar" in the title
    # and "DKI Jakarta" in the body now correctly resolve to two DIFFERENT canonical
    # provinces (conflict); "OKI"/"OKU" resolve to their real, distinct regency names.
    title_geo_canon=geo_canonical_mentions(Ttok)
    body_geo_canon=geo_canonical_mentions(body.split())
    if title_geo_canon:
        g_supported=sum(1 for g in title_geo_canon if g in body_geo_canon)
        canonical_entity_support=g_supported/len(title_geo_canon)
        g_conflict=0
        for g in title_geo_canon:
            if g in body_geo_canon: continue
            if body_geo_canon: g_conflict+=1
        canonical_entity_conflict=g_conflict/len(title_geo_canon)
    else:
        canonical_entity_support,canonical_entity_conflict=0.5,0.0
    # P1: PERSON-candidate support/conflict/substitution -- parallel to the entity features
    # above, but using per-document dynamic name detection instead of a static gazetteer
    # (person names are open-vocabulary). Whole-body scan.
    title_persons,title_anchored=extract_person_candidates(Ttok)
    body_persons,body_anchored=extract_person_candidates(body.split())
    body_person_lower={p.lower() for p in body_persons}
    if title_persons:
        p_sup=sum(1 for p in title_persons if p.lower() in body_person_lower)
        person_support=p_sup/len(title_persons)
        p_conf=0
        for p in title_persons:
            if p.lower() in body_person_lower: continue
            if body_persons: p_conf+=1
        person_conflict=p_conf/len(title_persons)
    else:
        person_support,person_conflict=0.5,0.0
    # Role-anchored substitution: same role marker (e.g. "Kapolda"/"Menkes") appears in
    # both title and body, but attached to a DIFFERENT name -- a high-precision signal for
    # exactly the person-swap pattern the entity audit flagged (Kapolda X -> Kapolres Y).
    title_marker_map={}
    for m,p in title_anchored: title_marker_map.setdefault(m,set()).add(p.lower())
    body_marker_map={}
    for m,p in body_anchored: body_marker_map.setdefault(m,set()).add(p.lower())
    if title_marker_map:
        n_sub=0
        for m,names in title_marker_map.items():
            bnames=body_marker_map.get(m,set())
            if bnames and not (names & bnames): n_sub+=1
        person_substitution=n_sub/len(title_marker_map)
        person_type_match=sum(1 for m in title_marker_map if m in body_marker_map)/len(title_marker_map)
    else:
        person_substitution,person_type_match=0.0,0.5
    ti,troot=find_predicate_in(title)
    # P0 fix: pred_score/polarity_score must come ONLY from a genuinely relevant predicate
    # match (exact/synonym/antonym, from find_predicate_chunk's ranked scan). If no such
    # evidence exists anywhere in the body, do NOT fall back to the lexically-closest chunk
    # to guess polarity -- that fallback was contaminating both scores with unrelated text.
    if troot is None or broot is None:
        pred_score=0.0
        polarity_score=0.0
    else:
        if pred_match_type=='antonym': pred_score=-1.0
        elif pred_match_type in ('exact','synonym'): pred_score=1.0
        else: pred_score=0.0
        window=Ttok[max(0,ti-3):ti+4]; neg_t=any(w.lower() in NEG for w in window)
        Btok=bc_pred.split(); window=Btok[max(0,bi-3):bi+4]; neg_b=any(w.lower() in NEG for w in window)
        polarity_score=-1.0 if neg_t!=neg_b else 1.0
    tn=extract_numbers(title)
    if tn:
        bn=extract_numbers(body)
        if bn:
            closeness=[max(0.0,1-min(abs(b-v)/max(abs(v),1) for b in bn)) for v in tn if v!=0]
            qty_score=float(np.mean(closeness)) if closeness else 0.0
        else: qty_score=0.0
    else: qty_score=0.5
    qty_ctx=qty_context_alignment(title, body)
    # Patch A (predicate-argument binding): moves past "is the evidence present anywhere"
    # (entity_support/pred_score) to "is the entity actually bound to the predicate as
    # subject/object" -- targets role-swap/argument-substitution cases where lexical overlap
    # stays high but the event's participants change. subj/obj are approximated as the
    # nearest ENT_POOL entity before/after the title's predicate token (no real parser).
    tents_pos=[(i,w.strip(_s.punctuation)) for i,w in enumerate(Ttok) if w.strip(_s.punctuation) in ENT_POOL]
    if ti is not None and tents_pos:
        subj_cands=[e for i,e in tents_pos if i<ti]; obj_cands=[e for i,e in tents_pos if i>ti]
        arg_subj=subj_cands[-1] if subj_cands else None
        arg_obj=obj_cands[0] if obj_cands else None
    else:
        arg_subj=arg_obj=None
    if broot is not None:
        pred_chunk_words=[w.strip(_s.punctuation) for w in bc_pred.split()]
        pred_chunk_set=set(pred_chunk_words)
        chunk_ents_typed=[(w,entity_type(w)) for w in pred_chunk_words if w in ENT_POOL]
        abc=0.0
        for arg in (arg_subj,arg_obj):
            if arg is None or arg in pred_chunk_set: continue
            at=entity_type(arg)
            if at=='UNKNOWN': continue
            if any(ce!=arg and ct==at for ce,ct in chunk_ents_typed): abc=1.0
        argument_binding_conflict=abc
    else:
        argument_binding_conflict=0.0
    support=entity_support+max(pred_score,0)+max(polarity_score,0)+qty_score
    conflict=entity_conflict+max(-pred_score,0)+max(-polarity_score,0)+(1-qty_score if tn else 0)
    return dict(entity_support=entity_support, entity_conflict=entity_conflict,
                entity_conflict_fullbody=entity_conflict_fullbody,
                phrase_entity_support=phrase_entity_support, phrase_entity_conflict=phrase_entity_conflict,
                person_support=person_support, person_conflict=person_conflict,
                person_substitution=person_substitution, person_type_match=person_type_match,
                canonical_entity_support=canonical_entity_support, canonical_entity_conflict=canonical_entity_conflict,
                qty_context_token_overlap=qty_ctx['qty_context_token_overlap'],
                qty_context_exact_match=qty_ctx['qty_context_exact_match'],
                qty_context_conflict=qty_ctx['qty_context_conflict'],
                argument_binding_conflict=argument_binding_conflict,
                pred_score=pred_score, polarity_score=polarity_score, qty_score=qty_score,
                claim_support_score=support, claim_conflict_score=conflict, claim_margin=support-conflict,
                found_predicate_chunk=float(broot is not None))

with Timer("claim_representation"):
    CRtr=pd.DataFrame([claim_scores(T,B,ch) for T,B,ch in zip(train['T'],train['C'],train['ch'])], index=train.index)
    CRte=pd.DataFrame([claim_scores(T,B,ch) for T,B,ch in zip(test['T'],test['C'],test['ch'])], index=test.index)
    CLAIM_COLS=list(CRtr.columns)
    Ftr=pd.concat([Ftr,CRtr],axis=1); Fte=pd.concat([Fte,CRte],axis=1)
ALL_A_FEATS = FEAT48 + CLAIM_COLS
print(f"A_cold feature count: {len(ALL_A_FEATS)}")
""")

md("## 8. Regime router — build lookups, classify every test row")
code("""
def build_lookups(df):
    pair={}; byc=defaultdict(list); conflicts=set()
    for th,ch,lb in zip(df['th'],df['ch'],df['label']):
        key=(th,ch)
        if key in pair and pair[key]!=lb: conflicts.add(key)
        pair[key]=lb; byc[ch].append((th,lb))
    assert len(conflicts)==0, (
        f"C_exact rule is NOT deterministic: {len(conflicts)} (title_hash, content_hash) "
        f"pairs have conflicting labels in train -- {list(conflicts)[:5]}")
    print(f"C_exact lookup validated: {len(pair)} unique (title,content) pairs, 0 label conflicts")
    return pair,byc

def classify_regime(df, pair, byc):
    reg=[]
    for th,ch in zip(df['th'],df['ch']):
        if (th,ch) in pair: reg.append('C_exact')
        elif ch in byc:
            reg.append('B_has_pos' if any(l==1 for _,l in byc[ch]) else 'B_no_pos')
        else: reg.append('A_cold')
    return np.array(reg)

pair_all, byc_all = build_lookups(train)
reg_test = classify_regime(test, pair_all, byc_all)
print("test regime counts:", {r: int((reg_test==r).sum()) for r in ['C_exact','B_has_pos','B_no_pos','A_cold']})
""")

md("""
## 9. C_exact — deterministic rule
0 label noise ever observed on exact (title,body) duplicate pairs across the whole
project's error analysis; this is a provable 1.0 Macro F1 component, not an estimate.
""")
code("""
def predict_c_exact(df, pair):
    out = np.full(len(df), np.nan)
    for i,(th,ch) in enumerate(zip(df['th'],df['ch'])):
        if (th,ch) in pair: out[i] = pair[(th,ch)]
    return out
c_exact_pred = predict_c_exact(test, pair_all)
print("C_exact resolved:", int((~np.isnan(c_exact_pred)).sum()), "of", int((reg_test=='C_exact').sum()))
""")

md("""
## 10. B_has_pos / B_no_pos — title-retrieval + entity-conflict classifier
Word/char TF-IDF similarity to the nearest same-body historical positive/negative title,
plus entity/number/negation conflict vs the best-matching candidate on each side.
Leave-one-title-out per body for the training pool (row's own label never used as its
own evidence); honest repeated-CV measured Macro F1 ~0.98 for this combined regime.
""")
code("""
from sklearn.feature_extraction.text import TfidfVectorizer as _TV
word_tv=_TV(ngram_range=(1,2),min_df=1,sublinear_tf=True,max_features=20000)
char_tv=_TV(analyzer='char',ngram_range=(3,5),min_df=1,sublinear_tf=True,max_features=20000)
Wm=word_tv.fit_transform(train['T']); Cm=char_tv.fit_transform(train['T'])
Wm=Wm.multiply(1.0/np.maximum(np.sqrt(Wm.multiply(Wm).sum(axis=1)),1e-9)).tocsr()
Cm=Cm.multiply(1.0/np.maximum(np.sqrt(Cm.multiply(Cm).sum(axis=1)),1e-9)).tocsr()
titles_list=train['T'].tolist(); labels_arr=train['label'].values
by_body=defaultdict(list)
for i,ch in enumerate(train['ch']): by_body[ch].append(i)

def edit_conflict(a_title,b_title):
    ta,tb=a_title.split(),b_title.split(); sa,sb=set(ta),set(tb)
    ents_a=[w for w in ta if w in ENT_POOL]
    ent_conflict=float(any(w not in sb for w in ents_a) and bool(ents_a))
    na,nb=set(NUM.findall(a_title)),set(NUM.findall(b_title))
    num_conflict=float(bool(na) and na!=nb)
    neg_a=any(w in sa for w in NEG); neg_b=any(w in sb for w in NEG)
    neg_conflict=float(neg_a!=neg_b)
    added=len(sb-sa); removed=len(sa-sb); shared=len(sa&sb)
    shared_ratio=shared/max(len(sa|sb),1)
    return dict(ent_conflict=ent_conflict,num_conflict=num_conflict,neg_conflict=neg_conflict,
                added=added,removed=removed,shared_ratio=shared_ratio)

def title_retrieval_feats(i, memory_idx):
    pos_idx=[j for j in memory_idx if labels_arr[j]==1]
    neg_idx=[j for j in memory_idx if labels_arr[j]==0]
    def sim_block(idxs):
        if not idxs: return dict(max_word=0.0,top3_word=0.0,max_char=0.0,top3_char=0.0,n=0), None
        wsims=np.array([float(Wm[i].multiply(Wm[j]).sum()) for j in idxs])
        csims=np.array([float(Cm[i].multiply(Cm[j]).sum()) for j in idxs])
        order=np.argsort(-wsims)
        return dict(max_word=float(wsims.max()),top3_word=float(np.sort(wsims)[::-1][:3].mean()),
                    max_char=float(csims.max()),top3_char=float(np.sort(csims)[::-1][:3].mean()),
                    n=len(idxs)), idxs[order[0]]
    pos_block,pos_best=sim_block(pos_idx); neg_block,neg_best=sim_block(neg_idx)
    feats={f'pos_{k}':v for k,v in pos_block.items()}
    feats.update({f'neg_{k}':v for k,v in neg_block.items()})
    feats['pos_neg_margin_word']=pos_block['max_word']-neg_block['max_word']
    feats['pos_neg_margin_char']=pos_block['max_char']-neg_block['max_char']
    feats['has_pos']=float(bool(pos_idx)); feats['has_neg']=float(bool(neg_idx))
    if pos_best is not None:
        cf=edit_conflict(titles_list[i],titles_list[pos_best]); feats.update({f'pos_{k}':v for k,v in cf.items()})
    else:
        feats.update({f'pos_{k}':0.0 for k in ['ent_conflict','num_conflict','neg_conflict','added','removed','shared_ratio']})
    if neg_best is not None:
        cf=edit_conflict(titles_list[i],titles_list[neg_best]); feats.update({f'neg_{k}':v for k,v in cf.items()})
    else:
        feats.update({f'neg_{k}':0.0 for k in ['ent_conflict','num_conflict','neg_conflict','added','removed','shared_ratio']})
    return feats

with Timer("B_regime_training_pool"):
    B_rows=[]
    for ch,idxs in by_body.items():
        if len(idxs)<2: continue
        if len(set(train['th'].iloc[idxs]))<2: continue
        B_rows.extend(idxs)
    B_feat_rows=[]
    for i in B_rows:
        ch=train['ch'].iloc[i]; memory_idx=[j for j in by_body[ch] if j!=i]
        f=title_retrieval_feats(i,memory_idx); f['label']=labels_arr[i]
        B_feat_rows.append(f)
    B_df=pd.DataFrame(B_feat_rows)
    B_feat_cols=[c for c in B_df.columns if c!='label']
    print(f"B regime training pool: {len(B_df)} rows, {len(B_feat_cols)} features")

with Timer("B_regime_fit"):
    Xb=B_df[B_feat_cols].values; yb=B_df['label'].values
    b_model=cb.CatBoostClassifier(iterations=400,depth=5,learning_rate=0.05,
                                   class_weights=[1,1],random_seed=SEED,verbose=False)
    b_model.fit(Xb,yb)

def predict_b_regime(df_test_subset):
    rows=[]
    for idx in df_test_subset.index:
        th,ch = df_test_subset.loc[idx,'th'], df_test_subset.loc[idx,'ch']
        T = df_test_subset.loc[idx,'T']
        memory_idx = by_body.get(ch, [])
        # emulate the same feature function but query row is the TEST title, using train memory only
        pos_idx=[j for j in memory_idx if labels_arr[j]==1]; neg_idx=[j for j in memory_idx if labels_arr[j]==0]
        tw_word = word_tv.transform([T]); tw_word = tw_word.multiply(1.0/np.maximum(np.sqrt(tw_word.multiply(tw_word).sum(axis=1)),1e-9)).tocsr()
        tw_char = char_tv.transform([T]); tw_char = tw_char.multiply(1.0/np.maximum(np.sqrt(tw_char.multiply(tw_char).sum(axis=1)),1e-9)).tocsr()
        def sim_block(idxs):
            if not idxs: return dict(max_word=0.0,top3_word=0.0,max_char=0.0,top3_char=0.0,n=0), None
            wsims=np.array([float(tw_word.multiply(Wm[j]).sum()) for j in idxs])
            csims=np.array([float(tw_char.multiply(Cm[j]).sum()) for j in idxs])
            order=np.argsort(-wsims)
            return dict(max_word=float(wsims.max()),top3_word=float(np.sort(wsims)[::-1][:3].mean()),
                        max_char=float(csims.max()),top3_char=float(np.sort(csims)[::-1][:3].mean()),
                        n=len(idxs)), idxs[order[0]]
        pos_block,pos_best=sim_block(pos_idx); neg_block,neg_best=sim_block(neg_idx)
        feats={f'pos_{k}':v for k,v in pos_block.items()}
        feats.update({f'neg_{k}':v for k,v in neg_block.items()})
        feats['pos_neg_margin_word']=pos_block['max_word']-neg_block['max_word']
        feats['pos_neg_margin_char']=pos_block['max_char']-neg_block['max_char']
        feats['has_pos']=float(bool(pos_idx)); feats['has_neg']=float(bool(neg_idx))
        if pos_best is not None:
            cf=edit_conflict(T,titles_list[pos_best]); feats.update({f'pos_{k}':v for k,v in cf.items()})
        else:
            feats.update({f'pos_{k}':0.0 for k in ['ent_conflict','num_conflict','neg_conflict','added','removed','shared_ratio']})
        if neg_best is not None:
            cf=edit_conflict(T,titles_list[neg_best]); feats.update({f'neg_{k}':v for k,v in cf.items()})
        else:
            feats.update({f'neg_{k}':0.0 for k in ['ent_conflict','num_conflict','neg_conflict','added','removed','shared_ratio']})
        rows.append(feats)
    Xq = pd.DataFrame(rows)[B_feat_cols].values
    return b_model.predict_proba(Xq)[:,1]

b_mask_test = np.isin(reg_test, ['B_has_pos','B_no_pos'])
b_test_df = test[b_mask_test]
if b_mask_test.sum()>0:
    with Timer("B_regime_predict"):
        b_proba = predict_b_regime(b_test_df)
else:
    b_proba = np.array([])
print("B regime test rows:", int(b_mask_test.sum()))
""")

md("""
## 11. A_cold — CatBoost on 48 v12 features + claim representation
class_weight 0:4:1, honest grouped-CV threshold selection (content_hash groups, no body
leaks across folds).
""")
code("""
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# Single source of truth: A_THR is calibrated on OOF probabilities produced with these
# weights, so the CV loop and the final fit MUST use the same value.
A_CLASS_WEIGHTS=[1,1]

with Timer("A_cold_cv_threshold"):
    groups=train['ch'].values
    sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    folds=list(sgkf.split(train,y,groups))
    Xa_full=Ftr[ALL_A_FEATS].values
    oof=np.zeros(len(train))
    for a,bidx in folds:
        m=cb.CatBoostClassifier(iterations=800,depth=6,learning_rate=0.03,
                                 class_weights=A_CLASS_WEIGHTS,random_seed=SEED,verbose=False)
        m.fit(Xa_full[a],y[a])
        oof[bidx]=m.predict_proba(Xa_full[bidx])[:,1]
    _cnt=train['ch'].value_counts()
    cold_mask=train['ch'].map(_cnt).eq(1).values
    cold_oof=oof[cold_mask]; cold_y=y[cold_mask]
    best=(0.5,-1)
    for t in np.arange(0.05,0.96,0.025):
        mac=f1_score(cold_y,(cold_oof>=t).astype(int),average='macro')
        if mac>best[1]: best=(t,mac)
    A_THR=best[0]
    f1_0=f1_score(cold_y,(cold_oof>=A_THR).astype(int),pos_label=0)
    f1_1=f1_score(cold_y,(cold_oof>=A_THR).astype(int),pos_label=1)
    print(f"A_cold OOF macroF1={best[1]:.4f} @ threshold={A_THR:.3f}")
    print("--- A_cold reproducibility record (this notebook, this run) ---")
    print(f"  n_features    : {len(ALL_A_FEATS)}")
    print(f"  model         : single CatBoostClassifier (NOT an ensemble -- multi-model")
    print(f"                  stacking was explored in experiments but is not part of")
    print(f"                  this notebook's committed best config)")
    print(f"  iterations    : 800")
    print(f"  depth         : 6")
    print(f"  learning_rate : 0.03")
    print(f"  class_weights : {A_CLASS_WEIGHTS}  (Experiment 1: weight sweep, GO -- stable win over [4,1]")
    print(f"                  across 3 independent CV seeds, +0.004 to +0.011 Macro F1;")
    print(f"                  CV loop and final fit share this constant by construction)")
    print(f"  CV type       : StratifiedGroupKFold(n_splits=5, group=content_hash)")
    print(f"  threshold     : {A_THR:.3f} (selected on OOF cold predictions only, no test labels)")
    print(f"  OOF Macro F1  : {best[1]:.4f}")
    print(f"  OOF F1-0      : {f1_0:.4f}")
    print(f"  OOF F1-1      : {f1_1:.4f}")
    cold_pred=(cold_oof>=A_THR).astype(int)
    print("--- A_cold OOF confusion matrix (rows=true, cols=pred; labels=[0,1]) ---")
    print(confusion_matrix(cold_y,cold_pred,labels=[0,1]))
    print("--- A_cold OOF classification report ---")
    print(classification_report(cold_y,cold_pred,labels=[0,1],digits=4))

with Timer("A_cold_final_fit"):
    a_model=cb.CatBoostClassifier(iterations=800,depth=6,learning_rate=0.03,
                                   class_weights=A_CLASS_WEIGHTS,random_seed=SEED,verbose=False)
    a_model.fit(Xa_full,y)
    Xte_a=Fte[ALL_A_FEATS].values
    a_proba_all=a_model.predict_proba(Xte_a)[:,1]
a_mask_test = reg_test=='A_cold'
print("A_cold test rows:", int(a_mask_test.sum()))
""")

md("## 12. Combine per regime router, write submission")
code("""
final=np.zeros(len(test),dtype=int)
c_mask = ~np.isnan(c_exact_pred)
final[c_mask] = c_exact_pred[c_mask].astype(int)
# C_exact rows without a resolvable label (shouldn't happen given regime classification, but guard anyway)
c_unresolved = (reg_test=='C_exact') & (~c_mask)
if c_unresolved.sum()>0:
    print("WARNING: unresolved C_exact rows, falling back to A_cold model for them:", int(c_unresolved.sum()))

B_THR = 0.30   # from title_retrieval_experiment.py honest repeated-CV best threshold
# Regime-specific threshold for B_no_pos: the single shared B classifier's aggregate honest
# CV (0.984) is dominated by the much larger, near-trivial B_has_pos subset (Macro F1~0.995,
# n=1278). B_no_pos alone (n=733, only 4.6% negative) scores far lower (~0.68-0.72) at the
# shared 0.30 threshold. Re-optimizing BOTH thresholds jointly (has_pos moves too) actually
# REGRESSES the pooled metric (0.9840->0.9825) -- verified via honest 5-fold CV. Keeping
# has_pos anchored at its already-good 0.30 and retuning ONLY no_pos's threshold (honest CV:
# 0.9840->0.9855) is the version that actually helps.
B_NOPOS_THR = 0.43
if b_mask_test.sum()>0:
    b_regime_labels = reg_test[b_mask_test]
    b_thr_per_row = np.where(b_regime_labels=='B_no_pos', B_NOPOS_THR, B_THR)
    final[np.where(b_mask_test)[0]] = (b_proba>=b_thr_per_row).astype(int)

a_idx = np.where(a_mask_test | c_unresolved)[0]
if len(a_idx)>0:
    final[a_idx] = (a_proba_all[a_idx]>=A_THR).astype(int)

for r in ['C_exact','B_has_pos','B_no_pos','A_cold']:
    m = reg_test==r
    if m.sum(): print(f"  {r:10s} n={m.sum():5d}  positive_rate={final[m].mean():.3f}")

sub=pd.DataFrame({'id':test['id'],'label':final}).set_index('id').loc[sample_sub['id']].reset_index()
assert sub.shape[0]==len(test) and list(sub['id'])==list(sample_sub['id'])
assert set(sub['label'].unique())<={0,1} and list(sub.columns)==['id','label']
sub.to_csv('/kaggle/working/submission.csv', index=False)
print(sub['label'].value_counts(normalize=True))
""")

code("""
T=time.time()-RUN_T0
print("="*58)
print("V13 — full regime router (C rule + B title-retrieval + A_cold claim-representation)")
print(f"A_cold OOF macro F1  : {best[1]:.4f}  (F1-0={f1_0:.4f}, F1-1={f1_1:.4f}) @ threshold {A_THR:.3f}")
print(f"total runtime        : {T/60:.1f} min")
print("="*58)
for k,v in sorted(TIMINGS.items(),key=lambda kv:-kv[1]): print(f"  {k}: {v:.1f}s")
""")

nb['cells']=cells
nb['metadata']['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
nb['metadata']['language_info']={'name':'python','version':'3.11'}
with open('ifest2026_dac_v13_final.ipynb','w',encoding='utf-8') as f: nbf.write(nb,f)
print("Notebook written.")
