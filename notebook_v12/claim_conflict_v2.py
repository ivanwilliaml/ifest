"""Claim Conflict Layer v2 for A_cold, precision-first design per human audit findings:
~50% entity substitution, ~8-10% negation/contradiction, ~4% number, ~1-2% action.

Stage 1: typed entity gazetteers (LOCATION / ORGANIZATION / PERSON-residual) replacing the
         old capitalized-word-only pool that caught noise (Gubernur/COVID/Pemprov).
Stage 2: entity-role conflict v2 (same-type substitution only: missing title LOCATION vs
         alternative LOCATION present in the most-relevant body chunk -- not any entity).
Stage 3: number v2 (parses "500 000"/"1,5 juta"/etc, magnitude-tolerant matching instead of
         exact-string, fixing the concrete regex bug found in the manual audit).
Stage 4: local-scope negation (window around the best-matching chunk, not document-wide).
Each channel is tested standalone first (precision/recall), then combined, via the same
grouped 5-fold CV on the true 48-feature v12 baseline.
"""
import os, sys, re, time, math, string as _s, unicodedata, hashlib
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score
import catboost as cb

SEED=42
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

CACHE = r"D:\Lomba\IFEST2026_DAC\notebook_v12\_diag_cache"
Ftr = pd.read_csv(os.path.join(CACHE,'features.csv'))
train = pd.read_csv(os.path.join(CACHE,'train_slim.csv'))
y = train['label'].values
FEAT48 = [c for c in Ftr.columns if c != 'action_conflict']

URL=re.compile(r'https?://\S+|www\.\S+'); HTML=re.compile(r'<[^>]+>'); WS=re.compile(r'\s+')
def clean(s):
    s=unicodedata.normalize('NFKC',str(s)); s=HTML.sub(' ',s); s=URL.sub(' ',s)
    return WS.sub(' ',s).strip()
train['T']=train['title'].apply(clean); train['C']=train['content'].apply(clean)
titles=train['T'].tolist(); bodies=train['C'].tolist()
def nh(x):
    x=unicodedata.normalize('NFKC',str(x)).lower(); return re.sub(r'\s+',' ',x).strip()
train['ch']=train['content'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
_cnt=train['ch'].value_counts()
cold_mask = train['ch'].map(_cnt).eq(1).values
cold_idx = np.where(cold_mask)[0]; cy = y[cold_idx]
log(f"cold rows: {cold_mask.sum()}")

groups=train['ch'].values
sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
folds=list(sgkf.split(train,y,groups))

# =====================================================================
# STAGE 1: typed gazetteers
# =====================================================================
LOCATIONS = {
    # provinces / common short forms
    'Jakarta','DKI','Jabar','Jateng','Jatim','Jabodetabek','Bandung','Surabaya','Semarang',
    'Yogyakarta','Yogya','DIY','Bali','Sumut','Sumbar','Sumsel','Riau','Jambi','Bengkulu',
    'Lampung','Kalbar','Kalteng','Kalsel','Kaltim','Kaltara','Sulut','Sulteng','Sulsel',
    'Sultra','Gorontalo','Maluku','Papua','NTB','NTT','Banten','Aceh','Kepri','Babel',
    'Bogor','Depok','Bekasi','Tangerang','Cirebon','Tasikmalaya','Sukabumi','Cianjur',
    'Garut','Kuningan','Banyuwangi','Malang','Kediri','Sidoarjo','Gresik','Solo','Semarang',
    'Karanganyar','Sragen','Mojokerto','Jember','Pasuruan','Lamongan','Banjarnegara',
    'Cilacap','Kudus','Jepara','Sleman','Bantul','Trenggalek','Cianjur','Cirebon',
    'Makassar','Medan','Palembang','Batam','Balikpapan','Samarinda','Manado','Pontianak',
    'Banjarmasin','Denpasar','Mataram','Kupang','Ambon','Jayapura','Kendari','Palu',
    # countries
    'Indonesia','Malaysia','Singapura','Thailand','Filipina','Vietnam','China','Tiongkok',
    'Jepang','Korea','Taiwan','India','Pakistan','Australia','Zimbabwe','Kenya',
    'Amerika','AS','Inggris','Prancis','Jerman','Italia','Spanyol','Portugal','Belanda',
    'Rusia','Brasil','Kanada','Meksiko','Mesir','Arab','Turki','Iran','Irak','Swedia',
    'Norwegia','Denmark','Swiss','Austria','Belgia','Yunani','Polandia','Ukraina',
    'Selandia','Wuhan','Shanghai','Beijing',
}
ORGANIZATIONS = {
    'BPOM','Kemenkes','Kominfo','Kemendagri','Kemenag','Kemendikbud','Kemdikbud','BNPB',
    'BPBD','Satpol','Polri','TNI','DPR','DPRD','KPK','MPR','MUI','IDI','KPU','Kejaksaan',
    'Golkar','PDIP','PPP','PAN','PKS','PKB','Demokrat','Nasdem','Gerindra','Perindo',
    'WHO','UNICEF','PBB','Pfizer','Moderna','Sinovac','AstraZeneca','BioNTech',
    'Satgas','BUMN','KAI','Garuda','Pertamina','PLN','BNI','BI',
}

def type_of(word):
    if word in LOCATIONS: return 'LOCATION'
    if word in ORGANIZATIONS: return 'ORG'
    return 'PERSON'  # residual: anything else in the purity-mined pool

# reuse purity-mined pool (proper nouns) as the raw entity candidate set
up,lo=Counter(),Counter()
for t in train['C']:
    for w in t.split():
        c=w.strip(_s.punctuation)
        if not c.isalpha() or len(c)<3: continue
        if c[:1].isupper(): up[c]+=1
        else: lo[c.capitalize()]+=1
pur={w:up[w]/(up[w]+lo.get(w,0)) for w in up if up[w]>=5}
STOPWORDS_ROLE = {'Gubernur','Wagub','Presiden','Wapres','Menteri','Menkes','Menko','Kementerian',
    'Ketua','Wali','Bupati','Wakil','Kepala','Dinas','Badan','Komisi','Partai','Kapolri',
    'Kapolres','Kapolda','Menkeu','Mendagri','Mensesneg','Kabareskrim','Sekretaris','Direktur',
    'Komisioner','Anggota','Pemerintah','Kasatgas','Karo','Kabid','Kadin','Kadis','Pemprov',
    'Pemkot','Pemkab','COVID','Corona','Covid','Menristek','Panglima','Jaksa','Hakim','Rektor',
    'Camat','Lurah','Kanwil'}
ENT_POOL = {w for w,p in pur.items() if p>=0.85 and w not in STOPWORDS_ROLE} | LOCATIONS | ORGANIZATIONS
log(f"typed entity pool: {len(ENT_POOL)}  (LOCATION={len(LOCATIONS)} ORG={len(ORGANIZATIONS)})")

def toks(s): return re.findall(r"[\w']+", s.lower())
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
def best_chunk(title, chunks):
    tw=set(toks(title)); best_c, best_score = chunks[0], -1
    for c in chunks:
        score=len(tw & set(toks(c)))
        if score>best_score: best_score=score; best_c=c
    return best_c

# =====================================================================
# STAGE 3: number v2 -- proper normalization + magnitude-tolerant matching
# =====================================================================
NUM_TOKEN = re.compile(r'\d+(?:[.,]\d+)?')
MULT = {'ribu':1e3,'juta':1e6,'miliar':1e9,'milyar':1e9,'triliun':1e12}
def extract_numbers(text):
    """Return list of normalized float values, handling space-separated thousands
    (e.g. '500 000' -> 500000) and juta/ribu/miliar multipliers."""
    words = text.split()
    vals=[]
    i=0
    while i < len(words):
        w = words[i].strip(_s.punctuation)
        m = NUM_TOKEN.fullmatch(w)
        if m:
            num_str = w.replace(',','.')
            try: val = float(num_str)
            except: val=None
            if val is not None:
                j=i+1
                # merge consecutive space-separated integer groups: "500 000" -> 500000
                while j < len(words):
                    w2 = words[j].strip(_s.punctuation)
                    if re.fullmatch(r'\d{3}', w2) and val==int(val):
                        val = val*1000 + int(w2); j+=1
                    else: break
                # check for trailing multiplier word
                if j < len(words):
                    w3 = words[j].lower().strip(_s.punctuation)
                    if w3 in MULT: val *= MULT[w3]; j+=1
                vals.append(val)
                i=j; continue
        i+=1
    return vals

def number_conflict_v2(title, body):
    tn = extract_numbers(title)
    if not tn: return 0.0, 0.0  # (conflict, n_title_numbers)
    bn = extract_numbers(body)
    if not bn: return 0.0, float(len(tn))
    n_conflict=0
    for v in tn:
        if v==0: continue
        rel_diffs = [abs(b-v)/max(abs(v),1) for b in bn]
        if min(rel_diffs) > 0.1:  # no body number within 10% relative tolerance
            n_conflict += 1
    frac_conflict = n_conflict/len(tn)
    return float(frac_conflict>0.5), float(len(tn))  # majority of title numbers unsupported

# =====================================================================
# STAGE 2: entity-role conflict v2 (typed, same-type substitution only)
# =====================================================================
def entity_role_v2(title, body):
    Ttok = title.split()
    tents = [(w, type_of(w)) for w in Ttok if w in ENT_POOL]
    if not tents: return 0.0, 0.0
    craw = {w.strip(_s.punctuation) for w in body.split()}
    bc = best_chunk(title, chunks_of(body))
    chunk_ents = [(w.strip(_s.punctuation), type_of(w.strip(_s.punctuation)))
                  for w in bc.split() if w.strip(_s.punctuation) in ENT_POOL]
    n_missing=0; n_typed_conflict=0
    for e, etype in tents:
        if e in craw: continue
        n_missing += 1
        # same-type alternative present in the best-matching chunk?
        if any(ce!=e and ct==etype for ce,ct in chunk_ents):
            n_typed_conflict += 1
    return float(n_typed_conflict>0), float(len(tents))

# =====================================================================
# STAGE 4: local-scope negation (window around best-matching chunk)
# =====================================================================
NEG = ['tidak','tak','bukan','belum','tanpa','gagal','ditolak','membantah','bantah',
       'sangkal','menyangkal','menolak']
def negation_local_v2(title, body):
    tset = set(w.lower() for w in title.split())
    neg_t = any(w in tset for w in NEG)
    bc = best_chunk(title, chunks_of(body))
    bset = set(w.lower() for w in bc.split())
    neg_b_local = any(w in bset for w in NEG)
    return float(neg_t != neg_b_local)

log("computing claim-conflict-v2 features for all rows...")
rows=[]
for T,B in zip(titles,bodies):
    er_conf, n_ent = entity_role_v2(T,B)
    num_conf, n_num = number_conflict_v2(T,B)
    neg_conf = negation_local_v2(T,B)
    rows.append(dict(entity_role_v2=er_conf, n_typed_entities=n_ent,
                      number_conflict_v2=num_conf, n_title_numbers=n_num,
                      negation_local_v2=neg_conf))
CCdf = pd.DataFrame(rows)
print(CCdf.describe().round(4))

# reuse existing action_conflict from cache (antonym lexicon, already computed)
CCdf['action_conflict'] = Ftr['action_conflict'] if 'action_conflict' in Ftr.columns else pd.read_csv(os.path.join(CACHE,'features.csv'))['action_conflict']

def make_cat(): return cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                                              class_weights=[4,1], random_seed=SEED, verbose=False)
def run_cv(Xdf):
    oof=np.zeros(len(train)); Xv=Xdf.values
    for a,b in folds:
        m=make_cat(); m.fit(Xv[a],y[a]); oof[b]=m.predict_proba(Xv[b])[:,1]
    return oof
def best_macro(p, yy):
    best=(0.5,-1)
    for t in np.arange(0.05,0.96,0.025):
        mac=f1_score(yy,(p>=t).astype(int),average='macro')
        if mac>best[1]: best=(t,mac)
    return best
def report(oof, name):
    co=oof[cold_idx]; t,mac=best_macro(co,cy)
    pred=(co>=t).astype(int); f1c=f1_score(cy,pred,average=None,labels=[0,1])
    p0=precision_score(cy,pred,pos_label=0,zero_division=0); r0=recall_score(cy,pred,pos_label=0,zero_division=0)
    print(f"{name:40s} macroF1={mac:.4f} F1_0={f1c[0]:.4f} F1_1={f1c[1]:.4f} P0={p0:.4f} R0={r0:.4f}")
    return mac

log("=== standalone channel precision/recall (rule-only, no model, threshold at flag=1) ===")
for col in ['entity_role_v2','number_conflict_v2','negation_local_v2','action_conflict']:
    flag = CCdf[col].values.astype(int)
    pred = 1 - flag  # flag fires -> predict 0
    p0 = precision_score(y[cold_idx], pred[cold_idx], pos_label=0, zero_division=0)
    r0 = recall_score(y[cold_idx], pred[cold_idx], pos_label=0, zero_division=0)
    f10 = f1_score(y[cold_idx], pred[cold_idx], pos_label=0, zero_division=0)
    n_fire = flag[cold_idx].sum()
    print(f"  {col:22s} fires={n_fire:4d}  precision0={p0:.4f}  recall0={r0:.4f}  F1_0={f10:.4f}")

log("=== staged ablation (added to true 48-feature baseline) ===")
oof_base = run_cv(Ftr[FEAT48]); report(oof_base, "true baseline (48 feats)")

Ftr_e = pd.concat([Ftr[FEAT48], CCdf[['entity_role_v2','n_typed_entities']]], axis=1)
report(run_cv(Ftr_e), "+ entity_role_v2 (typed)")

Ftr_n = pd.concat([Ftr[FEAT48], CCdf[['number_conflict_v2']]], axis=1)
report(run_cv(Ftr_n), "+ number_conflict_v2")

Ftr_g = pd.concat([Ftr[FEAT48], CCdf[['negation_local_v2']]], axis=1)
report(run_cv(Ftr_g), "+ negation_local_v2")

Ftr_all = pd.concat([Ftr[FEAT48], CCdf[['entity_role_v2','n_typed_entities','number_conflict_v2','negation_local_v2']]], axis=1)
report(run_cv(Ftr_all), "+ ALL claim-conflict-v2 combined")

print(f"\nreference: prior best (48-feat + BM25-sentence + class_weight 4:1): macroF1=0.6586 F1_0=0.3540")
log(f"total runtime: {(time.time()-T0)/60:.1f} min")
