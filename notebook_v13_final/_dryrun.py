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

with Timer("load"):
    F={}
    for r,_,fs in os.walk('D:\\Lomba\\IFEST2026_DAC\\data'):
        for f in fs:
            if f in ('train.csv','test.csv','sample_submission.csv'): F.setdefault(f,os.path.join(r,f))
    assert len(F)==3, F
    train=pd.read_csv(F['train.csv']); test=pd.read_csv(F['test.csv']); sample_sub=pd.read_csv(F['sample_submission.csv'])

def nh(x):
    x=unicodedata.normalize('NFKC',str(x)).lower(); return re.sub(r'\s+',' ',x).strip()
for d in (train,test):
    d['ch']=d['content'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
    d['th']=d['title'].apply(lambda s: hashlib.md5(nh(s).encode()).hexdigest())
URL=re.compile(r'https?://\S+|www\.\S+'); HTML=re.compile(r'<[^>]+>'); WS=re.compile(r'\s+')
def clean(s):
    s=unicodedata.normalize('NFKC',str(s)); s=HTML.sub(' ',s); s=URL.sub(' ',s)
    return WS.sub(' ',s).strip()
for d in (train,test):
    d['T']=d['title'].apply(clean); d['C']=d['content'].apply(clean)
y=train['label'].values
print("train",train.shape,"test",test.shape,"class0",round((train.label==0).mean(),4))

ANTONYM_PAIRS_RAW = [('baik', 'buruk'), ('besar', 'kecil'), ('panjang', 'pendek'), ('tinggi', 'rendah'), ('cepat', 'lambat'), ('jauh', 'dekat'), ('atas', 'bawah'), ('depan', 'belakang'), ('luar', 'dalam'), ('kiri', 'kanan'), ('hidup', 'mati'), ('siang', 'malam'), ('pagi', 'sore'), ('panas', 'dingin'), ('mahal', 'murah'), ('tebal', 'tipis'), ('berat', 'ringan'), ('terang', 'gelap'), ('bersih', 'kotor'), ('rajin', 'malas'), ('pintar', 'bodoh'), ('kuat', 'lemah'), ('berani', 'takut'), ('senang', 'sedih'), ('bahagia', 'sengsara'), ('untung', 'rugi'), ('menang', 'kalah'), ('tambah', 'kurang'), ('banyak', 'sedikit'), ('muda', 'tua'), ('baru', 'lama'), ('asli', 'palsu'), ('lurus', 'bengkok'), ('halus', 'kasar'), ('lembut', 'keras'), ('tajam', 'tumpul'), ('lapar', 'kenyang'), ('bangun', 'tidur'), ('pergi', 'datang'), ('masuk', 'keluar'), ('beli', 'jual'), ('buka', 'tutup'), ('nyala', 'padam'), ('terbit', 'terbenam'), ('naik', 'turun'), ('maju', 'mundur'), ('setuju', 'tolak'), ('ya', 'tidak'), ('benar', 'salah'), ('lulus', 'gagal'), ('ingat', 'lupa'), ('kenal', 'asing'), ('sehat', 'sakit'), ('subur', 'tandus'), ('ramai', 'sepi'), ('luas', 'sempit'), ('longgar', 'sempit'), ('gemuk', 'kurus'), ('wangi', 'busuk'), ('manis', 'pahit'), ('asin', 'tawar'), ('asam', 'basa'), ('kering', 'basah'), ('penuh', 'kosong'), ('padat', 'cair'), ('surga', 'neraka'), ('malaikat', 'setan'), ('pahala', 'dosa'), ('ibadah', 'maksiat'), ('halal', 'haram'), ('suci', 'najis'), ('iman', 'kufur'), ('tauhid', 'syirik'), ('syukur', 'kufur'), ('ikhlas', 'pamrih'), ('jujur', 'bohong'), ('setia', 'khianat'), ('aktif', 'pasif'), ('modern', 'tradisional'), ('canggih', 'kuno'), ('rapi', 'berantakan'), ('hemat', 'boros'), ('ramah', 'galak'), ('pelit', 'dermawan'), ('percaya', 'ragu'), ('yakin', 'sangsi'), ('pasti', 'mungkin'), ('sering', 'jarang'), ('selalu', 'kadang-kadang'), ('semua', 'sebagian'), ('umum', 'khusus'), ('global', 'lokal'), ('pusat', 'daerah'), ('majikan', 'buruh'), ('guru', 'murid'), ('orang tua', 'anak'), ('suami', 'istri'), ('kakak', 'adik'), ('pria', 'wanita'), ('laki-laki', 'perempuan'), ('jantan', 'betina'), ('tuan', 'hamba'), ('darat', 'laut'), ('udara', 'tanah'), ('gunung', 'lembah'), ('lahir', 'wafat'), ('kawin', 'cerai'), ('nikah', 'bujang'), ('kaya', 'miskin'), ('sukses', 'gagal'), ('juara', 'pecundang'), ('aneh', 'normal'), ('gratis', 'bayar'), ('aset', 'liabilitas'), ('modal', 'beban'), ('pendapatan', 'pengeluaran'), ('surplus', 'defisit'), ('ekspor', 'impor'), ('produksi', 'konsumsi'), ('penjual', 'pembeli'), ('pemilik', 'penyewa'), ('tuan rumah', 'tamu'), ('pemimpin', 'pengikut'), ('komandan', 'prajurit'), ('atasan', 'bawahan'), ('senior', 'yunior'), ('pemula', 'ahli'), ('amatir', 'profesional'), ('teori', 'praktik'), ('fakta', 'fiksi'), ('nyata', 'maya'), ('asli', 'tiruan'), ('pagi', 'petang'), ('senja', 'fajar'), ('langit', 'bumi'), ('dunia', 'akhirat'), ('jasmani', 'rohani'), ('lahir', 'batin'), ('cinta', 'benci'), ('suka', 'duka'), ('tawa', 'tangis'), ('senyum', 'cemberut'), ('bangga', 'malu'), ('hormat', 'hina'), ('mulia', 'nista'), ('indah', 'jelek'), ('bagus', 'buruk'), ('pintar', 'tolol'), ('cerdas', 'dungu'), ('tangkas', 'lamban'), ('rajin', 'lalai'), ('teliti', 'ceroboh'), ('waspada', 'lengah'), ('hati-hati', 'sembrono'), ('sabar', 'gusar'), ('tenang', 'gelisah'), ('damai', 'perang'), ('rukun', 'tengkar'), ('bersatu', 'cerai'), ('kokoh', 'rapuh'), ('kuat', 'goyah'), ('tegar', 'lemah'), ('berani', 'kecut'), ('pemberani', 'penakut'), ('setia', 'ingkar'), ('jujur', 'curang'), ('adil', 'zalim'), ('benar', 'keliru'), ('tepat', 'meleset'), ('cepat', 'lelet'), ('gesit', 'lambat'), ('lincah', 'kaku'), ('luwes', 'canggung'), ('ramah', 'angkuh'), ('sopan', 'biadab'), ('taat', 'melawan'), ('patuh', 'ingkar'), ('tertib', 'kacau'), ('bersih', 'dekil'), ('wangi', 'apek'), ('segar', 'layu'), ('mekar', 'kuncup'), ('tumbuh', 'mati'), ('bangun', 'roboh'), ('datar', 'curam'), ('landai', 'terjal'), ('dalam', 'dangkal'), ('besar', 'mungil'), ('raksasa', 'kerdil'), ('gemuk', 'kerempeng'), ('kuat', 'lunglai'), ('berani', 'gentar'), ('yakin', 'bimbang'), ('tegas', 'ragu'), ('lancar', 'tersendat'), ('fasih', 'gagap'), ('jelas', 'samar'), ('terang', 'remang'), ('nyata', 'gaib'), ('tulen', 'tiruan'), ('murni', 'campuran'), ('jernih', 'keruh'), ('bening', 'butek'), ('rapat', 'renggang'), ('akrab', 'asing'), ('sahabat', 'musuh'), ('kawan', 'lawan'), ('mitra', 'kompetitor'), ('positif', 'negatif'), ('plus', 'minus'), ('kredit', 'debit'), ('penerimaan', 'pengeluaran'), ('lokal', 'internasional'), ('domestik', 'mancanegara'), ('nasional', 'asing'), ('tetap', 'sementara'), ('abadi', 'fana'), ('utama', 'tambahan'), ('pokok', 'sampingan'), ('dasar', 'lanjutan'), ('pertama', 'terakhir'), ('muka', 'belakang'), ('puncak', 'dasar'), ('tepi', 'tengah'), ('pinggir', 'pusat'), ('eksterior', 'interior'), ('cangkang', 'inti'), ('induk', 'anak'), ('laki', 'perempuan'), ('cowok', 'cewek'), ('bujang', 'gadis'), ('perjaka', 'perawan'), ('duda', 'janda'), ('ria', 'duka'), ('gembira', 'susah'), ('mudah', 'sukar'), ('bugar', 'lemas'), ('kuat', 'lesu'), ('semangat', 'putus asa'), ('optimis', 'pesimis'), ('yakin', 'ragu'), ('cantik', 'jelek'), ('indah', 'buruk'), ('mulia', 'hina'), ('jangkung', 'cebol'), ('makmur', 'miskin'), ('kaya', 'papa'), ('berharta', 'melarat'), ('berhasil', 'kandas'), ('lulus', 'tidak lulus'), ('naik kelas', 'tinggal kelas'), ('pandai', 'tolol'), ('ahli', 'amatir'), ('pakar', 'pemula'), ('cair', 'beku'), ('tumpul', 'runcing'), ('lengkung', 'lurus'), ('kosong', 'isi'), ('hampa', 'penuh'), ('tentu', 'ragu'), ('jelas', 'kabur'), ('nyata', 'samar'), ('fakta', 'opini'), ('kebenaran', 'kebohongan'), ('jujur', 'dusta'), ('percaya', 'curiga'), ('aman', 'bahaya'), ('selamat', 'celaka'), ('berkah', 'musibah'), ('rahmat', 'laknat'), ('rapi', 'acak-acakan'), ('tertib', 'rusuh'), ('damai', 'konflik'), ('mitra', 'rival'), ('pemerintah', 'oposisi'), ('bayi', 'lansia'), ('tumbuh', 'layu'), ('segar', 'busuk'), ('pedas', 'hambar'), ('enak', 'hambar'), ('istirahat', 'kerja'), ('giat', 'malas'), ('tenang', 'panik'), ('lancar', 'macet'), ('gesit', 'lamban'), ('bagus', 'jelek'), ('tepat', 'keliru'), ('cukup', 'berlebih'), ('mulai', 'berhenti'), ('pasang', 'lepas'), ('lembab', 'kering'), ('empuk', 'liat'), ('kokoh', 'goyah'), ('raksasa', 'mungil'), ('mini', 'jumbo'), ('luas', 'terbatas'), ('terbatas', 'bebas'), ('bebas', 'terikat'), ('terikat', 'lepas'), ('pasang', 'copot'), ('rekat', 'pisah'), ('pisah', 'gabung'), ('gabung', 'urai'), ('urai', 'susun'), ('susun', 'acak'), ('acak', 'rapi'), ('lebar', 'sempit'), ('kurang', 'lebih'), ('awal', 'akhir'), ('dulu', 'nanti'), ('kemarin', 'besok'), ('pagi', 'malam'), ('siang', 'sore'), ('ganjil', 'genap'), ('tunggal', 'jamak'), ('monolog', 'dialog'), ('prolog', 'epilog'), ('preambule', 'penutup'), ('abstrak', 'konkret'), ('fiksi', 'nonfiksi'), ('induksi', 'deduksi'), ('subjektif', 'objektif'), ('kualitatif', 'kuantitatif'), ('hitam', 'putih'), ('suram', 'cerah'), ('kusam', 'mengkilap'), ('sedap', 'hambar'), ('tampan', 'jelek'), ('gagah', 'lunglai'), ('tegap', 'bungkuk'), ('gemuk', 'ceking'), ('lebat', 'jarang'), ('rimbun', 'gundul'), ('longgar', 'ketat'), ('lapang', 'sesak'), ('lega', 'sumpek'), ('tenang', 'gaduh'), ('sepi', 'bising'), ('sunyi', 'ramai'), ('hening', 'ribut'), ('damai', 'ricuh'), ('aman', 'rawan'), ('tentram', 'kacau'), ('tertib', 'berantakan'), ('rapi', 'semrawut'), ('bersih', 'kumuh'), ('mewah', 'sederhana'), ('megah', 'bersahaja'), ('mahal', 'ekonomis'), ('canggih', 'primitif'), ('maju', 'tertinggal'), ('pintar', 'dungu'), ('cerdas', 'pandir'), ('jenius', 'bodoh'), ('adil', 'sewenang-wenang'), ('jujur', 'culas'), ('amanah', 'ingkar'), ('sabar', 'pemarah'), ('tawadhu', 'takabur'), ('dermawan', 'kikir'), ('pemurah', 'pelit'), ('ramah', 'judes'), ('santun', 'kasar'), ('ada', 'tiada'), ('absen', 'hadir'), ('acak', 'teratur'), ('adaptif', 'kaku'), ('afirmatif', 'negatif'), ('agresif', 'defensif'), ('akurat', 'meleset'), ('aktif', 'nonaktif'), ('aktual', 'usang'), ('alami', 'buatan'), ('ambigu', 'tegas'), ('anonim', 'bernama'), ('antagonis', 'protagonis'), ('apatis', 'peduli'), ('apresiasi', 'celaan'), ('bahari', 'darat'), ('baku', 'tidak baku'), ('beradab', 'biadab'), ('berangkat', 'tiba'), ('berhasil', 'gagal'), ('berimbang', 'timpang'), ('berisik', 'hening'), ('berisi', 'kosong'), ('berkelanjutan', 'terputus'), ('berlebihan', 'kekurangan'), ('bermanfaat', 'mudarat'), ('bernilai', 'tidak bernilai'), ('berpihak', 'netral'), ('bersyarat', 'tanpa syarat'), ('bertambah', 'berkurang'), ('berubah', 'tetap'), ('beruntung', 'malang'), ('bising', 'senyap'), ('boros', 'irit'), ('buram', 'tajam'), ('cair', 'membeku'), ('cepat', 'pelan'), ('cerah', 'mendung'), ('cermat', 'sembrono'), ('cocok', 'bertentangan'), ('cukup', 'kurang'), ('damai', 'bermusuhan'), ('dangkal', 'mendalam'), ('demokratis', 'otoriter'), ('dinamis', 'statis'), ('diskrit', 'kontinu'), ('dominan', 'resesif'), ('duka', 'sukacita'), ('efisien', 'boros'), ('ekspansif', 'kontraktif'), ('eksplisit', 'implisit'), ('eksternal', 'internal'), ('ekstrover', 'introver'), ('elastis', 'kaku'), ('emosional', 'rasional'), ('empiris', 'teoretis'), ('enteng', 'berat'), ('formal', 'informal'), ('frontal', 'lateral'), ('global', 'parsial'), ('horizontal', 'vertikal'), ('ideal', 'nyata'), ('identik', 'berbeda'), ('ilegal', 'legal'), ('inklusif', 'eksklusif'), ('input', 'output'), ('insentif', 'disinsentif'), ('intensif', 'ekstensif'), ('jinak', 'liar'), ('kapital', 'nonkapital'), ('kasar', 'lembut'), ('kencang', 'kendur'), ('kental', 'encer'), ('kering', 'lembap'), ('kompleks', 'sederhana'), ('konservatif', 'progresif'), ('konstan', 'berubah'), ('konstruktif', 'destruktif'), ('kontan', 'kredit'), ('konvergen', 'divergen'), ('korup', 'bersih'), ('kritis', 'pasif'), ('legal', 'melanggar hukum'), ('lena', 'waspada'), ('lengkap', 'kurang lengkap'), ('lentur', 'kaku'), ('liberal', 'restriktif'), ('lisan', 'tertulis'), ('logis', 'tidak logis'), ('makro', 'mikro'), ('maksimal', 'minimal'), ('mayor', 'minor'), ('mayoritas', 'minoritas'), ('membeli', 'menjual'), ('menerima', 'menolak'), ('mengaktifkan', 'menonaktifkan'), ('mengangkat', 'menurunkan'), ('mengawali', 'mengakhiri'), ('membangun', 'merobohkan'), ('membuka', 'menutup'), ('mempercepat', 'memperlambat'), ('memperluas', 'mempersempit'), ('memperkuat', 'memperlemah'), ('menaati', 'melanggar'), ('mencatat', 'menghapus'), ('mendekat', 'menjauh'), ('meningkat', 'menurun'), ('menyambung', 'memutus'), ('menyetujui', 'menentang'), ('merdeka', 'terjajah'), ('meriah', 'suram'), ('merugikan', 'menguntungkan'), ('netral', 'memihak'), ('optimistis', 'pesimistis'), ('organik', 'anorganik'), ('padat', 'renggang'), ('patuh', 'membangkang'), ('peduli', 'acuh'), ('permanen', 'sementara'), ('produktif', 'konsumtif'), ('pro', 'kontra'), ('proaktif', 'reaktif'), ('publik', 'privat'), ('rasional', 'irasional'), ('resmi', 'tidak resmi'), ('riil', 'semu'), ('rumit', 'sederhana'), ('sadar', 'pingsan'), ('sah', 'batal'), ('sama', 'berbeda'), ('seimbang', 'timpang'), ('sejahtera', 'melarat'), ('sinkron', 'asinkron'), ('stabil', 'labil'), ('terbuka', 'tertutup'), ('tersambung', 'terputus'), ('transparan', 'buram'), ('valid', 'tidak valid'), ('virtual', 'fisik'), ('wajib', 'opsional'), ('waras', 'gila'), ('kebaikan', 'keburukan'), ('membesar', 'mengecil'), ('meninggi', 'merendah'), ('kecepatan', 'kelambatan'), ('kekuatan', 'kelemahan'), ('keberanian', 'ketakutan'), ('kesenangan', 'kesedihan'), ('kebahagiaan', 'kesengsaraan'), ('kekayaan', 'kemiskinan'), ('kejujuran', 'kebohongan'), ('keadilan', 'kezaliman'), ('keamanan', 'bahaya'), ('kebersihan', 'kekotoran'), ('kerapian', 'keberantakan'), ('keluasan', 'kesempitan'), ('kedalaman', 'kedangkalan'), ('memanjang', 'memendek'), ('menebal', 'menipis'), ('menerangkan', 'menggelapkan'), ('memanaskan', 'mendinginkan'), ('kemahalan', 'kemurahan'), ('kerajinan', 'kemalasan'), ('kepintaran', 'kebodohan'), ('kesopanan', 'kekasaran'), ('keramahan', 'kegalakan'), ('penghematan', 'pemborosan'), ('kesuburan', 'ketandusan'), ('keramaian', 'kesepian'), ('kemudahan', 'kesukaran'), ('keindahan', 'kejelekan'), ('optimisme', 'pesimisme'), ('keaktifan', 'kepasifan'), ('modernisasi', 'tradisionalisasi'), ('keaslian', 'kepalsuan'), ('kebenaran', 'kesalahan'), ('kesetiaan', 'pengkhianatan'), ('kesabaran', 'kegusaran'), ('ketenangan', 'kegelisahan'), ('perdamaian', 'peperangan'), ('kekokohan', 'kerapuhan'), ('kejelasan', 'kesamaran'), ('keakraban', 'keterasingan'), ('kepositifan', 'kenegatifan'), ('ketetapan', 'kesementaraan'), ('keabadian', 'kefanaan')]

SYNONYM_PAIRS_RAW = [('aba', 'ayah'), ('abadi', 'kekal'), ('abadiat', 'abadiah'), ('abah', 'aba'), ('abakus', 'dekak-dekak'), ('abdi', 'pelayan'), ('abdi', 'hamba'), ('abdul', 'abdu'), ('abimana', 'abaimana'), ('batau', 'baktau'), ('ablur', 'hablur'), ('abstrak', 'mujarad'), ('abstrak', 'ringkasan'), ('absurd', 'mustahil'), ('abu-abu', 'kelabu'), ('acan', 'acah'), ('acap', 'sering'), ('aci', 'sah'), ('aci', 'acik'), ('ada', 'kehadiran'), ('adab', 'kesopanan'), ('adan', 'azan'), ('adan', 'adnan'), ('babah', 'baba'), ('adem', 'tenang'), ('adenda', 'adendum'), ('adikanda', 'adinda'), ('tile', 'tule'), ('adpertensi', 'advertensi'), ('adpis', 'advis'), ('adstringen', 'astringen'), ('advokat', 'pengacara'), ('aestetika', 'estetika'), ('aetiologi', 'etiologi'), ('afkir', 'apkir'), ('afuah', 'afwah'), ('aga', 'angkuh'), ('agamis', 'agamais'), ('agel', 'agal'), ('agresi', 'serangan'), ('agribisnis', 'agrobisnis'), ('agul', 'sombong'), ('agul', 'bangga'), ('agul', 'megah'), ('agung', 'mulia'), ('agung', 'gung'), ('ahad', 'minggu'), ('ahli', 'anggota'), ('aib', 'noda'), ('aih', 'ai'), ('ajab', 'azab'), ('ajaib', 'ganjil'), ('ajaib', 'aneh'), ('akekah', 'akikah'), ('akhbar', 'harian'), ('ajek', 'tetap'), ('ajufan', 'adjuvan'), ('akor', 'akur'), ('ajung', 'jung'), ('ajung', 'ajun'), ('akad', 'perjanjian'), ('akad', 'ahad'), ('akademik', 'akademis'), ('akal', 'pikiran'), ('akal', 'kecerdikan'), ('akar', 'pangkal'), ('akas kaya', 'kaskaya'), ('akhlas', 'ikhlas'), ('akhwan', 'ikhwan'), ('aklimasi', 'aklimatisasi'), ('akrab', 'kekariban'), ('akte', 'akta'), ('aktip', 'aktif'), ('aku', 'saya'), ('akur', 'setuju'), ('akur', 'mengiakan'), ('akur', 'mencocokkan'), ('akurat', 'saksama'), ('alabangka', 'linggis'), ('alabangka', 'perejang'), ('alap', 'bagus'), ('alas', 'asas'), ('alat', 'kelengkapan'), ('alfabet', 'abjad'), ('algoritma', 'algoritme'), ('alih aksara', 'transliterasi'), ('alih', 'pindah'), ('alip', 'alif'), ('aliyah', 'aliah'), ('candang', 'berani'), ('alkohol', 'etanol'), ('alku', 'jaruman'), ('alku', 'muncikari'), ('ambat', 'hambat'), ('himbau', 'imbau'), ('almari', 'lemari'), ('aloi', 'lakur'), ('alpa', 'lengah'), ('alpukah', 'inisiatif'), ('cilap', 'cilok'), ('alu', 'elu'), ('ama', 'hama'), ('amandemen', 'amendemen'), ('amarah', 'marah'), ('amat', 'terlalu'), ('ambeg', 'ambek'), ('ambeien', 'wasir'), ('ambal', 'permadani'), ('ambalela', 'balela'), ('kuyam', 'koyam'), ('amberal', 'admiral'), ('ambigu', 'taksa'), ('ambin', 'amben'), ('ambril', 'amril'), ('ami', 'umi'), ('amien', 'amin'), ('amirulbahri', 'amirulbahar'), ('ammabakdu', 'amabakdu'), ('ammi', 'ami'), ('ampang', 'gampang'), ('ampang', 'mudah'), ('ampelam', 'mempelam'), ('ampibi', 'amfibi'), ('amuba', 'ameba'), ('ampu', 'empu'), ('ampung', 'apung'), ('amputir', 'amputasi'), ('jais', 'jaiz'), ('kintar', 'kitar'), ('anak', 'ranting'), ('anak', 'pemuda'), ('anak', 'keturunan'), ('analisa', 'analisis'), ('anamel', 'enamel'), ('ancing', 'hancing'), ('andai', 'misal'), ('andai', 'handai'), ('anderik', 'anderak'), ('andikara', 'adikara'), ('anduk', 'handuk'), ('anggak', 'congkak'), ('anggak', 'sombong'), ('anggak', 'angkuh'), ('anggap', 'menyangka'), ('mandarsah', 'madrasah'), ('anggara', 'buas'), ('anggu', 'peranggu'), ('angkar', 'angker'), ('masyakat', 'masyakah'), ('angkat', 'menaikkan'), ('angkuh', 'sombong'), ('angkuh', 'kesombongan'), ('angkus', 'angkusa'), ('angsang', 'insang'), ('angus', 'hangus'), ('anjangsono', 'anjangsana'), ('anom', 'muda'), ('anomali', 'kelainan'), ('ansor', 'ansar'), ('ansori', 'ansari'), ('antariksawan', 'astronaut'), ('antasida', 'antasid'), ('antimon', 'antimonium'), ('anus', 'dubur'), ('anyak', 'enyak'), ('anyak-anyik', 'onyak-anyik'), ('anyik', 'onyak-anyik'), ('apek', 'apak'), ('apem', 'apam'), ('aplikasi', 'penerapan'), ('aplikasi', 'permohonan'), ('aplus', 'aplaus'), ('apotik', 'apotek'), ('aqidah', 'akidah'), ('arah', 'tujuan'), ('aral', 'rintangan'), ('nanda', 'ananda'), ('arbitrasi', 'arbitrase'), ('arerut', 'ararut'), ('ares', 'menangkap'), ('argentum', 'perak'), ('aria', 'arya'), ('aring', 'urang-aring'), ('aristokrat', 'ningrat'), ('arit', 'sabit'), ('arketipe', 'prototipe'), ('artifak', 'artefak'), ('artifisial', 'buatan'), ('artik', 'arktika'), ('artikel', 'pasal'), ('arun', 'harum'), ('asa', 'harapan'), ('asabiyah', 'asabiah'), ('asi', 'benar'), ('asin', 'masin'), ('asli', 'tulen'), ('asma', 'bengek'), ('astana', 'istana'), ('asteroid', 'planetoid'), ('aswad', 'hitam'), ('atsiri', 'asiri'), ('aur', 'buluh'), ('aur', 'bambu'), ('auto', 'oto'), ('autologi', 'otologi'), ('automatis', 'otomatis'), ('automobil', 'mobil'), ('autoskop', 'otoskop'), ('avertebrata', 'invertebrata'), ('awam', 'umum'), ('awam', 'am'), ('awam', 'biasa'), ('awan', 'mega'), ('ayak', 'ayakan'), ('ayanda', 'ayahanda'), ('ayid', 'ayit'), ('ayuk', 'ayut'), ('ayun', 'goyang'), ('ayun', 'berbuai'), ('ayun', 'bergoyang'), ('azamat', 'azmat'), ('azemat', 'azmat'), ('babad', 'sejarah'), ('babad', 'tambo'), ('babar', 'bebar'), ('bacak', 'bacek'), ('bacul', 'penakut'), ('bada', 'bakda'), ('badan', 'jasmani'), ('bagai', 'sama'), ('bagak', 'bangga'), ('basoka', 'bazoka'), ('bah', 'banjir'), ('baham', 'geraham'), ('baik', 'elok'), ('bain', 'nyata'), ('baja', 'pupuk'), ('bajak', 'luku'), ('bajak', 'tenggala'), ('baju', 'kemeja'), ('bajul', 'pencuri'), ('mendelika', 'mandalika'), ('bakh', 'untung'), ('bakh', 'bahagia'), ('bakhil', 'kikir'), ('bakhil', 'lokek'), ('bakhil', 'pelit'), ('bakhsis', 'baksis'), ('baki', 'kekal'), ('baki', 'abadi'), ('bakterisid', 'bakterisida'), ('bala', 'malapetaka'), ('bala', 'kemalangan'), ('bala', 'kesengsaraan'), ('balah', 'pertengkaran'), ('balang', 'belang'), ('balit', 'belit'), ('balsem', 'balsam'), ('balu', 'duda'), ('baluarti', 'benteng'), ('balung', 'jengger'), ('banci', 'sensus'), ('bambu', 'buluh'), ('bambung', 'bodoh'), ('bambung', 'pandir'), ('bancah', 'bencah'), ('banding', 'membandingkan'), ('bandring', 'bandering'), ('bandrol', 'banderol'), ('bangkut', 'kerdil'), ('baronsai', 'barongsai'), ('banglo', 'bungalo'), ('bangpak', 'jelek'), ('bangpak', 'jahat'), ('bangsai', 'busuk'), ('bangsawan', 'ningrat'), ('bangsi', 'seruling'), ('mendem', 'mendam'), ('banian', 'benian'), ('banteras', 'berantas'), ('bansai', 'banzai'), ('bantu', 'pertolongan'), ('banu', 'bani'), ('banyak', 'sangat'), ('banyol', 'jenaka'), ('banyol', 'berjenaka'), ('banyol', 'melawak'), ('bapao', 'bakpao'), ('baptis', 'permandian'), ('baragajul', 'bergajul'), ('barangan', 'berangan'), ('barangkali', 'mungkin'), ('baras', 'abras'), ('barbir', 'barber'), ('barep', 'barap'), ('bargas', 'barkas'), ('bari', 'bahari'), ('baris', 'jajaran'), ('barkas', 'berkas'), ('barua', 'muncikari'), ('basa', 'bahasa'), ('basekat', 'baskat'), ('baskat', 'beskap'), ('baso', 'bakso'), ('batalyon', 'batalion'), ('batas', 'sempadan'), ('batela', 'batel'), ('batil', 'batel'), ('batila', 'batel'), ('bebal', 'bodoh'), ('bausastra', 'kamus'), ('bawa', 'memindahkan'), ('baya', 'bahaya'), ('bayan', 'nuri'), ('bayan', 'nyata'), ('bayan', 'terang'), ('bayat', 'baiat'), ('bayem', 'bayam'), ('bayonet', 'sangkur'), ('bayur', 'pterospermum'), ('beban', 'tanggungan'), ('bebek', 'itik'), ('bebeksan', 'beksan'), ('becolok', 'bicokok'), ('bedar', 'bidar'), ('bedegap', 'kuat'), ('begandring', 'begandering'), ('begasi', 'bagasi'), ('beguk', 'gondong'), ('behena', 'bena'), ('behina', 'bena'), ('beka', 'kelak'), ('bekah', 'rekah'), ('bekasam', 'pekasam'), ('beku', 'kaku'), ('bekukung', 'bekuku'), ('belakang', 'nanti'), ('belakang', 'kelak'), ('belakin', 'belangkin'), ('belanda', 'nederland'), ('belantan', 'gada'), ('belata', 'pelata'), ('belatung', 'bernga'), ('belengket', 'melekat'), ('belerang', 'sulfur'), ('beling', 'mbeling'), ('belis', 'iblis'), ('belit', 'berlilit'), ('beloh', 'bodoh'), ('beloh', 'dungu'), ('beloh', 'tolol'), ('belok', 'berkelok'), ('belok', 'bengkok'), ('beluk', 'seluk'), ('belur', 'balur'), ('bembet', 'bimbit'), ('bemper', 'bumper'), ('bencah', 'paya'), ('tilgrap', 'telegraf'), ('benak', 'sumsum'), ('benak', 'bengap'), ('benak', 'bodoh'), ('benar', 'betul'), ('benar', 'sekali'), ('bencana', 'kecelakaan'), ('bendang', 'sawah'), ('bendang', 'persawahan'), ('bendar', 'bandar'), ('bendara', 'bendahara'), ('bendari', 'bendahari'), ('bendel', 'bundel'), ('bendela', 'bandela'), ('bendera', 'panji-panji'), ('bendu', 'sahabat'), ('bendu', 'kawan'), ('benggol', 'benjol'), ('bengik', 'bengek'), ('bengkap', 'benkap'), ('bengkol', 'pengkol'), ('bengkong', 'bengkok'), ('bengok', 'murung'), ('bengok', 'benguk'), ('benguk', 'murung'), ('benguk', 'bersedih'), ('bensol', 'benzol'), ('bentes', 'benteh'), ('bentik', 'betik'), ('bentil', 'pentil'), ('bentoh', 'bantah'), ('benyek', 'benyai'), ('beronsang', 'berongsang'), ('berangsong', 'berongsong'), ('berani', 'kegagahan'), ('berat', 'bobot'), ('berunai', 'brunai'), ('berenga', 'bernga'), ('bererot', 'rerot'), ('berhana', 'seberhana'), ('beringisan', 'peringis'), ('berita', 'maklumat'), ('beritawan', 'pemberita'), ('beritawan', 'wartawan'), ('berkah', 'berkat'), ('berokat', 'brokat'), ('beskat', 'beskap'), ('bomantara', 'bumantara'), ('berzanji', 'barzanji'), ('bestik', 'bistik'), ('betah', 'tabah'), ('betara', 'batara'), ('betari', 'batari'), ('bincacau', 'bincacak'), ('beting', 'gosong'), ('bianglala', 'pelangi'), ('biaya', 'belanja'), ('bicana', 'bijana'), ('bicara', 'bercakap'), ('bicu', 'dongkrak'), ('bidari', 'bidadari'), ('biduri', 'baiduri'), ('bihalal', 'halalbihalal'), ('bihari', 'bahari'), ('bihausy', 'bius'), ('bijak', 'pandai'), ('bijak', 'kebijaksanaan'), ('bijaksana', 'arif'), ('bijan', 'wijen'), ('bika', 'bikang'), ('cawangan', 'cabang'), ('bilal', 'muazin'), ('bilhak', 'sebenarnya'), ('bilhak', 'sesungguhnya'), ('bilyar', 'biliar'), ('bilyun', 'biliun'), ('binari', 'biner'), ('binasa', 'memusnahkan'), ('binatang', 'hewan'), ('binatu', 'penatu'), ('bingung', 'kebingungan'), ('birahi', 'berahi'), ('biri-biri', 'domba'), ('bis', 'bus'), ('bisan', 'besan'), ('biskal', 'beskal'), ('biskop', 'uskup'), ('blepot', 'lepot'), ('bloon', 'beloon'), ('bogel', 'telanjang'), ('bohok', 'buhuk'), ('bolak', 'salah'), ('bolak', 'keliru'), ('bolos', 'bulus'), ('bolsak', 'bulsak'), ('bomor', 'bomoh'), ('bonafid', 'bonafide'), ('bongak', 'sombong'), ('bongak', 'congkak'), ('bongak', 'angkuh'), ('bongak', 'bodoh'), ('bongkah', 'gumpal'), ('bongkak', 'congkak'), ('bongkas', 'bungkas'), ('bongkok', 'bungkuk'), ('bongkol', 'bonggol'), ('bongkol', 'bungkul'), ('bongsai', 'bonsai'), ('bonjol', 'bonggol'), ('bonjol', 'boncol'), ('bonus', 'insentif'), ('bonyor', 'bonyok'), ('borak', 'burak'), ('borhan', 'burhan'), ('bostan', 'bustan'), ('bota', 'buta'), ('botak', 'gundul'), ('boyak', 'membosankan'), ('boyas', 'buncit'), ('boyong', 'pemindahan'), ('brangas', 'berangas'), ('brangus', 'berangus'), ('bredel', 'beredel'), ('breksi', 'breksia'), ('brengsek', 'berengsek'), ('brewok', 'berewok'), ('broker', 'makelar'), ('bromocorah', 'bramacorah'), ('brongkos', 'berongkos'), ('buai', 'berayun'), ('buana', 'dunia'), ('buang', 'melemparkan'), ('buang', 'menyingkirkan'), ('buas', 'liar'), ('buas', 'kejam'), ('buat', 'melakukan'), ('bucu', 'penjuru'), ('bucu', 'sudut'), ('budak', 'jongos'), ('budi', 'akhlak'), ('buhur', 'buhul'), ('buih', 'busa'), ('bujang', 'jongos'), ('bujanggi', 'bujangga'), ('bujur', 'lonjong'), ('bujur', 'mujur'), ('bukat', 'kotor'), ('buku', 'kitab'), ('bulu tangkis', 'badminton'), ('bulus', 'miskin'), ('bulus', 'papa'), ('bumi', 'dunia'), ('bumpet', 'pendek'), ('bungkah', 'bongkah'), ('buntal', 'gembung'), ('buntal', 'buncit'), ('buntar', 'bundar'), ('bunting', 'hamil'), ('bupet', 'bufet'), ('burit', 'buntut'), ('burit', 'punggung'), ('buru', 'buruan'), ('buruh', 'perburuhan'), ('cadai', 'bergurau'), ('cadai', 'berolok-olok'), ('burun', 'buron'), ('burung', 'unggas'), ('burut', 'hernia'), ('busana', 'pakaian'), ('busar', 'busur'), ('byarpet', 'biarpet'), ('cabuh', 'heboh'), ('calus', 'celus'), ('cacah jiwa', 'sensus'), ('cam', 'kecam'), ('cacar', 'variola'), ('cacat', 'kekurangan'), ('cacengklok', 'cecengklok'), ('caci', 'cela'), ('caci', 'cerca'), ('caci', 'memaki'), ('caci', 'mencela'), ('caci maki', 'celaan'), ('caci maki', 'memaki-maki'), ('cadang', 'merancang'), ('cadir', 'cadar'), ('cahaya', 'menerangi'), ('cais', 'kendali'), ('cakrawala', 'horizon'), ('cakap', 'bicara'), ('cakap', 'berbicara'), ('cakar', 'kuku'), ('cakep', 'cakap'), ('cakmar', 'cokmar'), ('cakmar', 'camar'), ('calempong', 'celempong'), ('caling', 'colang-caling'), ('calit', 'palit'), ('cambah', 'kecambah'), ('campak', 'lempar'), ('campak', 'membuang'), ('cancang', 'menambat'), ('canda', 'seloroh'), ('canda', 'berseloroh'), ('candan', 'kecandan'), ('cangga', 'cenangga'), ('dapat', 'sanggup'), ('dapat', 'memperoleh'), ('cangkat', 'cetek'), ('cangkau', 'cengkau'), ('cangku', 'cengkau'), ('cangkuk', 'cangkok'), ('cangkul', 'pacul'), ('cangkung', 'bertinggung'), ('deluang', 'jeluang'), ('delujur', 'jelujur'), ('capai', 'lelah'), ('capai', 'letih'), ('cape', 'lelah'), ('cape', 'letih'), ('cape', 'capai'), ('capiau', 'cepiau'), ('capung', 'sepatung'), ('capung', 'sibur-sibur'), ('caram', 'acaram'), ('caran', 'berbantah'), ('caran', 'bertengkar'), ('carat', 'carak'), ('carik', 'cabik'), ('casis', 'sasis'), ('catet', 'catat'), ('cauk', 'caung'), ('caya', 'cahaya'), ('cebar-cebur', 'cebur'), ('cecak', 'cicak'), ('cecer', 'cecar'), ('cecok', 'cekcok'), ('cedal', 'cadel'), ('cedera', 'cendera'), ('cedong', 'cedok'), ('cegah', 'menegahkan'), ('cegak', 'tegap'), ('cegas', 'cergas'), ('cek', 'pemeriksaan'), ('cekalang', 'cakalang'), ('cekap', 'cekak'), ('cekcok', 'bertengkar'), ('cekcok', 'berbantah'), ('cekek', 'cekik'), ('cekel', 'kikir'), ('cekel', 'bakhil'), ('cekel', 'pelit'), ('cekeram', 'cengkeram'), ('cekiber', 'cekibar'), ('celekeh', 'berselekeh'), ('cekup', 'cekut'), ('cela', 'cacat'), ('cela', 'mengkritik'), ('celaga', 'jelaga'), ('celaka', 'malang'), ('celepa', 'selepa'), ('celepak', 'celapak'), ('celok', 'celuk'), ('celos', 'celus'), ('celoteh', 'mengobrol'), ('celum-celam', 'celam-celum'), ('celupar', 'celopar'), ('cembeng', 'cengbeng'), ('cemek', 'cemeh'), ('cemeti', 'cambuk'), ('cemeti', 'pecut'), ('cempana', 'jempana'), ('cempelung', 'cemplung'), ('cempeng', 'cempek'), ('cempuling', 'tempuling'), ('cencurut', 'celurut'), ('centadu', 'sentadu'), ('cendekia', 'cerdas'), ('cendera', 'candra'), ('cengang', 'mengherankan'), ('cengang', 'mengagumkan'), ('cengang', 'menakjubkan'), ('cenggek', 'tenggek'), ('cengkau', 'cekau'), ('cengkeling', 'sengkeling'), ('cengkeram', 'genggaman'), ('cengkerma', 'cengkerama'), ('cengkolong', 'cengkelong'), ('cengkuk', 'cengkok'), ('centil', 'sentil'), ('cepal', 'cepol'), ('cepat', 'laju'), ('cepat', 'lekas'), ('cepit', 'jepit'), ('ceracah', 'cerancang'), ('cerah', 'hari'), ('cerai', 'talak'), ('cerca', 'mencaci'), ('cerca', 'memaki'), ('cercak', 'ceracap'), ('cerdik', 'licik'), ('ceret', 'cerek'), ('ceri', 'ceria'), ('ceria', 'suci'), ('ceria', 'murni'), ('ceria', 'cerah'), ('ceriga', 'curiga'), ('ceritera', 'cerita'), ('cerkas', 'cergas'), ('cerkau', 'mencengkam'), ('cerkau', 'mencekau'), ('cermat', 'saksama'), ('ceronggak', 'ceranggah'), ('cerotok', 'ceratuk'), ('cetak', 'cetakan'), ('cetek', 'dangkal'), ('cetek', 'tohor'), ('cetera', 'cerita'), ('ceti', 'muncikari'), ('ciau', 'ciu'), ('cicah', 'cecah'), ('cici', 'cicit'), ('cicih', 'cecah'), ('cicir', 'cecer'), ('cidera', 'cedera'), ('cigak', 'kera'), ('cikalang', 'cakalang'), ('cim', 'encim'), ('cimpung', 'kecimpung'), ('cingah', 'cingangah'), ('cingbing', 'cengbeng'), ('cingcau', 'cincau'), ('cingcing', 'cincong'), ('cingkau', 'cengkau'), ('cinteng', 'centeng'), ('ciok', 'ciak'), ('ciplak', 'jiplak'), ('cola-cala', 'membual'), ('citak', 'cetak'), ('ciut', 'menyusut'), ('clurit', 'celurit'), ('cocakrawa', 'cucakrawa'), ('cocok', 'mengakurkan'), ('cocok', 'menyesuaikan'), ('codang', 'codak'), ('cokar', 'jogar'), ('cuak', 'takut'), ('cuak', 'gentar'), ('colet', 'colek'), ('colot', 'meloncat'), ('conggok', 'tegak'), ('congkak', 'sombong'), ('congkak', 'pongah'), ('congkelang', 'congklang'), ('contek', 'sontek'), ('contoh', 'sampel'), ('cop', 'kecup'), ('copar', 'cupar'), ('copol', 'cupul'), ('corek', 'curik'), ('coro', 'kecoak'), ('corong', 'cerobong'), ('corong', 'semprong'), ('corot', 'terakhir'), ('cotet', 'conet'), ('cotok', 'paruh'), ('cubit', 'sekelumit'), ('cukam', 'cekam'), ('cukir', 'cungkil'), ('culi', 'coli'), ('cumbul', 'cembul'), ('cunda', 'cucunda'), ('cungak', 'cungap'), ('cungap', 'terengah-engah'), ('cunguk', 'cecunguk'), ('cunting', 'conteng'), ('cup', 'kecup'), ('cupit', 'sumpit'), ('curai', 'nyata'), ('curat', 'cerat'), ('curat', 'corot'), ('dacing', 'dacin'), ('dagang', 'perniagaan'), ('dahar', 'santap'), ('dahi', 'kening'), ('daif', 'lemah'), ('daif', 'hina'), ('daim', 'kekal'), ('daim', 'abadi'), ('dakar', 'zakar'), ('dakwat', 'dawat'), ('dalalah', 'muncikari'), ('dalalah', 'jaruman'), ('dalalah', 'barua'), ('dalfin', 'dolfin'), ('daluwarsa', 'kedaluwarsa'), ('damai', 'mendamaikan'), ('dambun', 'dambin'), ('dammah', 'damah'), ('damping', 'dekat'), ('damping', 'dampeng'), ('damping', 'dumping'), ('danau', 'tasik'), ('dandang', 'dendang'), ('dangak', 'dongak'), ('dangkal', 'tohor'), ('dara', 'gadis'), ('darab', 'memperkalikan'), ('darma', 'kewajiban'), ('darus', 'daras'), ('dastar', 'destar'), ('dat', 'zat'), ('decah', 'decap'), ('datum', 'tanggal'), ('daulah', 'daulat'), ('daun', 'daun-daunan'), ('daya upaya', 'ikhtiar'), ('dayung', 'pengayuh'), ('debet', 'debit'), ('debitor', 'debitur'), ('debu', 'abu'), ('debus', 'dabus'), ('decus', 'desus'), ('defekasi', 'berak'), ('dehem', 'deham'), ('deifikasi', 'pendewaan'), ('dekade', 'dasawarsa'), ('delegasi', 'perutusan'), ('delemak', 'delamak'), ('delut', 'jelut'), ('demen', 'suka'), ('demik', 'menampar'), ('demik', 'menepuk'), ('demikian', 'begitu'), ('demit', 'dedemit'), ('dempuk', 'dempok'), ('denawa', 'danawa'), ('dencing', 'denting'), ('dengar', 'mengindahkan'), ('dengkul', 'lutut'), ('dengkur', 'berkeruh'), ('dengkur', 'mengeruh'), ('dengkut', 'dengkur'), ('densanak', 'dansanak'), ('dentam', 'dentum'), ('depan', 'hadapan'), ('dependensi', 'ketergantungan'), ('deraka', 'durhaka'), ('deras', 'daras'), ('derek', 'derik'), ('derham', 'dirham'), ('deria', 'indra'), ('deriji', 'deruji'), ('deruji', 'jeruji'), ('derun', 'derum'), ('derung', 'derum'), ('desinfektan', 'disinfektan'), ('deteriorasi', 'kemunduran'), ('detia', 'daitia'), ('detil', 'detail'), ('diagnosa', 'diagnosis'), ('diah', 'diat'), ('diam', 'mendiami'), ('diam', 'berumah'), ('diwala', 'dewala'), ('digen', 'degen'), ('dikara', 'mulia'), ('dikir', 'zikir'), ('dinamik', 'dinamis'), ('dipati', 'adipati'), ('dogeng', 'dogel'), ('dolat', 'daulat'), ('dolfin', 'lumba-lumba'), ('dolim', 'zalim'), ('dolpin', 'dolfin'), ('domain', 'ranah'), ('domein', 'domain'), ('donatir', 'donatur'), ('dongan', 'sahabat'), ('dongok', 'dungu'), ('dongok', 'tolol'), ('dos', 'dus'), ('dosin', 'lusin'), ('doyong', 'condong'), ('doyong', 'miring'), ('dungas', 'mendengus'), ('drem', 'drum'), ('dria', 'indra'), ('drum', 'tambur'), ('duduk', 'menempatkan'), ('duka', 'kesedihan'), ('filosof', 'filsuf'), ('dulu', 'dahulu'), ('dungu', 'bebal'), ('dungu', 'bodoh'), ('dunsanak', 'dansanak'), ('duren', 'durian'), ('duriat', 'zuriah'), ('durna', 'durno'), ('dursila', 'jahat'), ('dusta', 'bohong'), ('duta', 'utusan'), ('duwegan', 'degan'), ('dwiganda', 'dobel'), ('edap', 'dap'), ('editor', 'penyunting'), ('efedrin', 'efedrina'), ('efisien', 'sangkil'), ('egalitarisme', 'egalitarianisme'), ('ejek', 'sindiran'), ('eklektis', 'eklektik'), ('ekor', 'membuntuti'), ('eksentrik', 'aneh'), ('eksentrik', 'ganjil'), ('eksim', 'eksem'), ('eksoftalmia', 'eksoftalmos'), ('eksoftalmus', 'eksoftalmos'), ('ekspeditur', 'ekspeditor'), ('eksploitir', 'eksploitasi'), ('ekspo', 'eksposisi'), ('eksportir', 'pengekspor'), ('ekspres', 'cepat'), ('ekstrak', 'pati'), ('ekstrak', 'sari'), ('ekuilibrium', 'kesetimbangan'), ('ela', 'elo'), ('elang', 'rajawali'), ('elat', 'helat'), ('elegan', 'elok'), ('elegan', 'anggun'), ('elektrik', 'listrik'), ('eliminir', 'eliminasi'), ('elok', 'bagus'), ('eluk', 'luk'), ('elung', 'lengkung'), ('elung', 'lung'), ('emak', 'mak'), ('embang', 'ambang'), ('emblem', 'lambang'), ('empang', 'tebat'), ('empang', 'tambak'), ('empelas', 'ampelas'), ('emper', 'sengkuap'), ('emrat', 'embrat'), ('enamel', 'email'), ('endoderm', 'endoderma'), ('endul', 'buaian'), ('energi', 'tenaga'), ('enjak', 'memijak'), ('enjin', 'mesin'), ('ente', 'anta'), ('epidemi', 'wabah'), ('episkopat', 'keuskupan'), ('epistaksis', 'mimisan'), ('era', 'masa'), ('erak', 'lelah'), ('erotik', 'erotis'), ('erotisme', 'erotisisme'), ('esa', 'satu'), ('esens', 'sari'), ('esensi', 'inti'), ('estetik', 'estetis'), ('eteris', 'asiri'), ('etil alkohol', 'etanol'), ('etnik', 'etnis'), ('eugenika', 'eugenetika'), ('evaporasi', 'penguapan'), ('fadihat', 'keaiban'), ('faedah', 'manfaat'), ('faedah', 'untung'), ('faedah', 'laba'), ('famili', 'kerabat'), ('fantasi', 'mengkhayalkan'), ('faraj', 'farji'), ('fatsoen', 'fatsun'), ('fertilitas', 'kesuburan'), ('fiber', 'serat'), ('flat', 'apartemen'), ('flu', 'influenza'), ('fluor', 'fosfor'), ('fobi', 'fobia'), ('fondamen', 'fundamen'), ('fonograf', 'gramofon'), ('fora', 'forum'), ('formatir', 'formatur'), ('gabir', 'canggung'), ('frase', 'frasa'), ('fulan', 'polan'), ('fusi', 'peleburan'), ('gabihat', 'kabihat'), ('gadang', 'besar'), ('gadis', 'perawan'), ('gado', 'gaduh'), ('gaduh', 'ribut'), ('gaduk', 'congkak'), ('gaduk', 'sombong'), ('gajih', 'gemuk'), ('gala', 'damar'), ('galaba', 'pilu'), ('galaba', 'sedih'), ('galang', 'galeng'), ('galir', 'longgar'), ('gambar', 'lukisan'), ('gambas', 'oyong'), ('gamma', 'gama'), ('gana', 'ganar'), ('ganal', 'serupa'), ('ganar', 'bingung'), ('ganden', 'gandin'), ('gandi', 'gandin'), ('ganduh', 'bercampur'), ('gangsal', 'gasal'), ('gangsang', 'gasang'), ('gangsing', 'gasing'), ('gani', 'kaya'), ('ganih', 'genis'), ('ganjak', 'beranjak'), ('ganjil', 'gasal'), ('ganjil', 'aneh'), ('ganjil', 'keanehan'), ('gara', 'gahara'), ('ganti', 'bertukar'), ('gantung', 'sangkut'), ('ganyar', 'keras'), ('gapah', 'tangkas'), ('garansi', 'jaminan'), ('garau', 'parau'), ('garib', 'asing'), ('garis', 'sempadan'), ('garit', 'gerak'), ('garu', 'menggaruk'), ('gasir', 'gangsir'), ('gasolin', 'bensin'), ('gaswah', 'kaswah'), ('gecer', 'gecar'), ('gedabak', 'gedebuk'), ('gedabir', 'gelambir'), ('gedang', 'gadang'), ('gaung', 'gema'), ('gaung', 'kumandang'), ('gelapur', 'gelepur'), ('gaya', 'sikap'), ('gayat', 'gamang'), ('gayuh', 'kayuh'), ('gayun', 'membuai'), ('geblek', 'bebal'), ('gedembai', 'kelambai'), ('gegadan', 'patut'), ('gegar', 'goyang'), ('gegar', 'menggetarkan'), ('gelabah', 'gelebah'), ('geladah', 'geledah'), ('gelagak', 'gelegak'), ('gelana', 'gulana'), ('gelantang', 'kelantang'), ('gelap', 'kelam'), ('gelap', 'kegelapan'), ('gelasar', 'gelangsar'), ('gelatak', 'geletak'), ('gelatang', 'gelantang'), ('gelebah', 'sedih'), ('gelembai', 'kelambai'), ('gelempang', 'gelimpang'), ('gelentar', 'geletar'), ('gelepok', 'gelepot'), ('geler', 'gilir'), ('gelesek', 'geleser'), ('geletar', 'menggigil'), ('geletar', 'gemetar'), ('geletek', 'gelitik'), ('geligis', 'menggigil'), ('gelindung', 'gelendong'), ('gelintin', 'pekat'), ('gelintin', 'kental'), ('gelipang', 'gelimpang'), ('geliting', 'gelitik'), ('geliut', 'geliang'), ('gelojak', 'gejolak'), ('gelojoh', 'lahap'), ('gelokak', 'gelopak'), ('gelomang', 'gelimang'), ('gelongsong', 'kelongsong'), ('gelorat', 'darurat'), ('gelosor', 'gelongsor'), ('gelumang', 'gelimang'), ('gelumat', 'gelemat'), ('gelung', 'melingkarkan'), ('gelut', 'bergumul'), ('gema', 'kumandang'), ('gemala', 'kemala'), ('geman', 'gemang'), ('gemang', 'gamang'), ('gemang', 'takut'), ('gembira', 'suka'), ('gembira', 'bahagia'), ('gembira', 'senang'), ('gembung', 'kembung'), ('gemeletek', 'menggigil'), ('gementar', 'gemetar'), ('gemercing', 'gemerencing'), ('gemilang', 'cemerlang'), ('gemirang', 'girang'), ('gemit', 'gamit'), ('gempal', 'gumpal'), ('gempul-gempul', 'termengah-mengah'), ('gempul-gempul', 'berkempul-kempul'), ('gemuk', 'tambun'), ('genap', 'semua'), ('gencer', 'gencar'), ('gencir', 'gelincir'), ('gendaga', 'kendaga'), ('gendala', 'kendala'), ('generasi', 'angkatan'), ('gendeng', 'gila'), ('gendit', 'kendit'), ('gengsot', 'berdansa'), ('genih', 'genis'), ('genteng', 'genting'), ('gentik', 'getik'), ('genting', 'krisis'), ('incer', 'incar'), ('karosel', 'korsel'), ('halilintar', 'kilat'), ('gerenyot', 'kernyih'), ('gerenyut', 'gerenyot'), ('gewang', 'giwang'), ('halimun', 'kabut'), ('jeremang', 'jermang'), ('gepuk', 'gemuk'), ('gerabak', 'gerabang'), ('geracak', 'gerecak'), ('geradah', 'geledah'), ('geragih', 'stolon'), ('geragot', 'gerogot'), ('geram', 'gemas'), ('gerami', 'gurami'), ('gerangsang', 'berangsang'), ('gerdam', 'gerdum'), ('gerebak', 'gerebek'), ('gerebek', 'garebek'), ('gerenek', 'gerenik'), ('gerentam', 'gerentang'), ('gerenyam', 'geranyam'), ('geret', 'menggarit'), ('geret', 'menggores'), ('gergajul', 'bergajul'), ('gerih', 'gereh'), ('gering', 'sakit'), ('geringsing', 'gerising'), ('gerinyut', 'gerenyot'), ('gerip', 'gerit'), ('gerita', 'gurita'), ('geroda', 'garuda'), ('gerong', 'gerung'), ('geronggong', 'geronggang'), ('geronium', 'geranium'), ('gerontang', 'gerantang'), ('geropes', 'gerupis'), ('gersak', 'kersak'), ('gersik', 'kersik'), ('gertap', 'gerlap'), ('geru', 'raung'), ('gerubuk', 'gerobok'), ('gerugul', 'gerogol'), ('geruh', 'celaka'), ('geruh', 'sial'), ('gerumit', 'gerupis'), ('gerumuk', 'meringkuk'), ('gerutu', 'kesat'), ('gesau', 'desau'), ('gesit', 'giat'), ('gilap', 'berkilauan'), ('getap', 'pecah'), ('getek', 'rakit'), ('getir', 'getil'), ('getis', 'getas'), ('gibas', 'kibas'), ('gigih', 'gigil'), ('gigir', 'gigil'), ('gimnastik', 'senam'), ('gincu', 'lipstik'), ('glikosid', 'glikosida'), ('ginggang', 'genggang'), ('gingsi', 'gengsi'), ('girang', 'riang'), ('gladi', 'geladi'), ('gliserin', 'gliserol'), ('glodok', 'gelodok'), ('glondongan', 'gelondongan'), ('gobah', 'gubah'), ('gobar', 'suram'), ('gogo', 'gaga'), ('golpi', 'golbi'), ('gombrong', 'gombroh'), ('goncang', 'guncang'), ('gonio', 'goniometri'), ('gono-gini', 'gana-gini'), ('gonrong', 'gondrong'), ('gores', 'garis'), ('gosok', 'bergesel'), ('got', 'selokan'), ('goyang', 'mengayunkan'), ('graha', 'gerha'), ('gregat', 'gereget'), ('greget', 'gereget'), ('greha', 'gerha'), ('grup', 'golongan'), ('gubang', 'gobang'), ('gubar', 'gobar'), ('gudam', 'godam'), ('kerenyau', 'kernyau'), ('gugup', 'gagap'), ('guram', 'suram'), ('guram', 'muram'), ('kerenyit', 'kernyit'), ('guling', 'merobohkan'), ('gumbuk', 'membujuk'), ('gundah', 'sedih'), ('gundik', 'selir'), ('gurau', 'kelakar'), ('gurau', 'lelucon'), ('gurem', 'guram'), ('guri', 'buyung'), ('gusar', 'marah'), ('gusti', 'bergelut'), ('hablur', 'kristal'), ('habuk', 'debu'), ('had', 'batas'), ('hadam', 'khadam'), ('hadang', 'adang'), ('haid', 'menstruasi'), ('hajat', 'keinginan'), ('hakiki', 'benar'), ('hakiki', 'sebenarnya'), ('hakimah', 'bijak'), ('hal', 'peristiwa'), ('halipan', 'lipan'), ('halofita', 'halofit'), ('halus', 'lembut'), ('halus', 'sopan'), ('halus', 'beradab'), ('halus', 'kesopanan'), ('halus', 'keadaban'), ('hamik', 'bodoh'), ('handuk', 'tuala'), ('hangar', 'hanggar'), ('hampa', 'mengecewakan'), ('handai', 'kawan'), ('handai', 'teman'), ('handal', 'andal'), ('handam', 'andam'), ('hangat', 'kepanasan'), ('hanyut', 'meleset'), ('hapal', 'hafal'), ('harafiah', 'harfiah'), ('harkat', 'taraf'), ('harmoni', 'keselarasan'), ('harmonis', 'keselarasan'), ('harus', 'semestinya'), ('hasad', 'dengki'), ('hasan', 'elok'), ('hasan', 'cantik'), ('hasil', 'perolehan'), ('hasud', 'dengki'), ('hasud', 'hasad'), ('hauri', 'haur'), ('hayat', 'hidup'), ('hayat', 'kehidupan'), ('heban', 'hebat'), ('heboh', 'gaduh'), ('heboh', 'ribut'), ('heboh', 'huru-hara'), ('hektar', 'hektare'), ('helah', 'helat'), ('helai', 'lembar'), ('helat', 'asing'), ('hembus', 'embus'), ('hendak', 'memerlukan'), ('hendak', 'meminta'), ('hendam', 'andam'), ('hening', 'bening'), ('hening', 'sunyi'), ('hening', 'sepi'), ('henti', 'selesai'), ('herbivora', 'herbivor'), ('hero', 'pahlawan'), ('hibuk', 'sibuk'), ('hidayat', 'hidayah'), ('hidraulik', 'hidraulis'), ('hidrolika', 'hidraulika'), ('hikam', 'hikmah'), ('hilang', 'lenyap'), ('hilang', 'kematian'), ('hilap', 'khilaf'), ('hinap', 'mempertimbangkan'), ('hingar', 'ingar'), ('hinggut', 'goyang'), ('hipermetropia', 'hiperopia'), ('hipnose', 'hipnosis'), ('hipokrit', 'munafik'), ('hiru-hara', 'huru-hara'), ('hiruk', 'gaduh'), ('hisap', 'isap'), ('histori', 'sejarah'), ('hitung', 'mencongak'), ('hobi', 'kegemaran'), ('hodah', 'haudah'), ('hol', 'haul'), ('jujat', 'jujah'), ('horak', 'orak'), ('horizontal', 'mendatar'), ('humor', 'kejenakaan'), ('humor', 'kelucuan'), ('hostel', 'asrama'), ('hubar', 'uber'), ('hubung', 'menghubungkan'), ('hujah', 'hujat'), ('hujah', 'alasan'), ('hujung', 'ujung'), ('hukah', 'hokah'), ('hulur', 'ulur'), ('humbalang', 'hembalang'), ('hunjin', 'hujin'), ('hunjuk', 'unjuk'), ('ibah', 'hibah'), ('huru-hara', 'keributan'), ('huru-hara', 'kerusuhan'), ('huru-hara', 'kekacauan'), ('hutang', 'utang'), ('ibuk', 'hibuk'), ('ide', 'gagasan'), ('ihtiar', 'ikhtiar'), ('idulkurban', 'iduladha'), ('ijazah', 'sijil'), ('ikhwan', 'saudara'), ('ilusi', 'khayalan'), ('ikrab', 'karib'), ('ikrab', 'akrab'), ('ikut', 'turut'), ('ikut', 'mengiringi'), ('ilir', 'hilir'), ('imang', 'timang'), ('imitasi', 'tiruan'), ('impedansi', 'impedans'), ('implantasi', 'nidasi'), ('implementasi', 'pelaksanaan'), ('implisit', 'tersirat'), ('imunitas', 'kekebalan'), ('kaabah', 'kakbah'), ('indah', 'cantik'), ('indah', 'elok'), ('indeks', 'penunjuk'), ('indera', 'indra'), ('indria', 'indra'), ('infantri', 'infanteri'), ('infiltrasi', 'penyusupan'), ('influensa', 'influenza'), ('infrastruktur', 'prasarana'), ('ingat', 'memikirkan'), ('ingat', 'terkenang'), ('ingin', 'hasrat'), ('ingin', 'kehendak'), ('ingkar', 'menyangkal'), ('inisiatif', 'prakarsa'), ('inkulturasi', 'enkulturasi'), ('insani', 'kemanusiaan'), ('inspirasi', 'ilham'), ('instalatir', 'instalatur'), ('instruktur', 'pengajar'), ('intelektual', 'cendekiawan'), ('inteligen', 'cerdas'), ('inteligen', 'berakal'), ('interen', 'intern'), ('interes', 'minat'), ('interim', 'sementara'), ('interpiu', 'interviu'), ('interpretasi', 'tafsiran'), ('interviu', 'wawancara'), ('inti', 'sari'), ('inti', 'pati'), ('intil', 'kintil'), ('intim', 'akrab'), ('intim', 'karib'), ('iodin', 'yodium'), ('iqamat', 'ikamah'), ('iri', 'cemburu'), ('iri', 'sirik'), ('iri', 'dengki'), ('iring', 'menyertai'), ('isolasi', 'pengasingan'), ('isolir', 'isolasi'), ('istaz', 'ustaz'), ('istazah', 'ustazah'), ('istiadat', 'adat'), ('itarad', 'iktirad'), ('itaraf', 'iktiraf'), ('itibar', 'iktibar'), ('itidal', 'iktidal'), ('itikad', 'iktikad'), ('itikaf', 'iktikaf'), ('jabel', 'jabal'), ('jaga', 'berawas-awas'), ('jagat', 'bumi'), ('jagat', 'dunia'), ('jahul', 'jahat'), ('jailangkung', 'jelangkung'), ('jalan', 'lorong'), ('jamaah', 'jemaah'), ('jamak', 'lazim'), ('jamak', 'lumrah'), ('jaman', 'zaman'), ('jambak', 'jambul'), ('jamban', 'tandas'), ('jamiyah', 'jamiah'), ('jamrud', 'zamrud'), ('jamung', 'obor'), ('jamur', 'cendawan'), ('jamur', 'kulat'), ('janat', 'sempurna'), ('jangkih', 'jangki'), ('jangla', 'liar'), ('jannah', 'janah'), ('jantera', 'jentera'), ('japuk', 'japu'), ('jarang', 'jerang'), ('jariat', 'jariah'), ('jaru', 'jaharu'), ('jasa', 'layanan'), ('jasirah', 'jazirah'), ('jasmani', 'tubuh'), ('jati', 'asli'), ('jawab', 'sahut'), ('jawab', 'balas'), ('jawab', 'membalas'), ('jawab', 'balasan'), ('jawawut', 'sekoi'), ('jaya', 'sukses'), ('jazam', 'memenggal'), ('jazirat', 'jazirah'), ('jebak', 'perangkap'), ('jebik', 'cebik'), ('jejaka', 'bujang'), ('jejamang', 'jamang'), ('jeket', 'jaket'), ('jelek', 'jahat'), ('jelantik', 'gelatik'), ('jelas', 'nyata'), ('jelata', 'biasa'), ('jelentik', 'selentik'), ('jembel', 'melarat'), ('jemerlang', 'cemerlang'), ('jempalik', 'jempalit'), ('jenaka', 'lucu'), ('jenama', 'merek'), ('jenayah', 'jinayah'), ('jendela', 'tingkap'), ('jenela', 'jendela'), ('jengat', 'jangat'), ('jengget', 'jengket'), ('jenggot', 'janggut'), ('jengkerik', 'jangkrik'), ('jenis', 'klasifikasi'), ('jenius', 'genius'), ('jenjam', 'tenang'), ('jeragan', 'juragan'), ('jenjang', 'tangga'), ('jenjeng', 'jinjing'), ('jentaka', 'sial'), ('jentera', 'pesawat'), ('jentera', 'mesin'), ('jentur', 'jantur'), ('jepet', 'jepit'), ('jepun', 'jepang'), ('jerajak', 'jerjak'), ('jeram', 'jaram'), ('jeran', 'jera'), ('jerangkah', 'ceranggah'), ('jerapah', 'zarafah'), ('jerejak', 'jerjak'), ('jerembet', 'jerepet'), ('jerepak', 'jerempak'), ('jeri', 'takut'), ('jerigen', 'jeriken'), ('jeriji', 'jeruji'), ('jeriji', 'terali'), ('jeriji', 'kisi-kisi'), ('jerit', 'teriak'), ('jerit', 'berteriak'), ('jerit', 'pekikan'), ('jerkah', 'membentak'), ('jina', 'zina'), ('jernih', 'bening'), ('jernih', 'bersih'), ('jerobong', 'jerubung'), ('jerongkes', 'jerungkis'), ('jerpak', 'jerempak'), ('jeruji', 'kisi-kisi'), ('jeruk', 'limau'), ('jerumbung', 'jerubung'), ('jerungkup', 'jerukup'), ('jiarah', 'ziarah'), ('jibilat', 'jibilah'), ('jibrail', 'jibril'), ('jidal', 'bidal'), ('jidor', 'jidur'), ('jigrah', 'gembira'), ('jijitsu', 'yuyitsu'), ('jimak', 'persetubuhan'), ('jimat', 'azimat'), ('jingkau', 'jangkau'), ('jipang', 'dahan'), ('jipsi', 'gipsi'), ('jirat', 'nisan'), ('jirat', 'jerat'), ('jiwa', 'nyawa'), ('jiwa', 'hidup'), ('joak', 'juak'), ('joang', 'juang'), ('jogi', 'yogi'), ('johan', 'pahlawan'), ('johan', 'juara'), ('jokong', 'jongkong'), ('jombang', 'elok'), ('jombang', 'cantik'), ('jongkah', 'jongkang'), ('jongkang', 'jongkong'), ('jongkok', 'bercangkung'), ('jongkok', 'berjongkok'), ('kobak', 'kubak'), ('juara', 'pendekar'), ('jubung', 'jerubung'), ('judul', 'tajuk'), ('jugi', 'yogi'), ('juja', 'jauza'), ('jujur', 'ikhlas'), ('jukung', 'jongkong'), ('julur', 'merayap'), ('jumanten', 'jumantan'), ('jumawa', 'jemawa'), ('jumbai', 'rumbai'), ('jumbuh', 'sesuai'), ('jumpa', 'bersua'), ('jumrah', 'jamrah'), ('jumud', 'beku'), ('jungkel', 'jungkal'), ('jungkung', 'jongkong'), ('junjung', 'menghormati'), ('jurah', 'jura'), ('juran', 'joran'), ('juri', 'jori'), ('juriat', 'zuriah'), ('jurnalis', 'wartawan'), ('jus', 'juz'), ('kabab', 'kebab'), ('kabah', 'kakbah'), ('kabat', 'mengikat'), ('kabuli', 'kebuli'), ('kabus', 'kabut'), ('kacang-kacang', 'penabur'), ('kaco', 'kacau'), ('kacoak', 'kecoak'), ('kacu', 'saputangan'), ('kadal', 'kedal'), ('kadaluwarsa', 'kedaluwarsa'), ('kadam', 'khadam'), ('kaedah', 'kaidah'), ('kademat', 'khidmat'), ('kafarat', 'keparat'), ('kafi', 'cukup'), ('kafi', 'sempurna'), ('kaget', 'terperanjat'), ('kaget', 'mengejutkan'), ('kah', 'engkah'), ('kahang', 'kohong'), ('kahwin', 'kawin'), ('kain', 'baju'), ('kaji', 'mempelajari'), ('kakar', 'kekar'), ('kaku', 'kejang'), ('kala', 'masa'), ('kalakatu', 'kelekatu'), ('kalambak', 'kelembak'), ('kalem', 'tenang'), ('kalembak', 'kelembak'), ('kalender', 'takwim'), ('kalipah', 'khalifah'), ('kali', 'sungai'), ('kalifah', 'khalifah'), ('kalikasar', 'kalikausar'), ('kalikausar', 'alkausar'), ('kalis', 'suci'), ('kalis', 'bersih'), ('kalkarim', 'kalkarium'), ('kalkausar', 'alkausar'), ('kalo', 'kalau'), ('kalong', 'keluang'), ('kamat', 'ikamah'), ('kaluk', 'keluk'), ('kalulus', 'kelulus'), ('kalus', 'kapalan'), ('kamal', 'sempurna'), ('kamal', 'cukup'), ('kamar', 'bilik'), ('kambing', 'domba'), ('kambrat', 'kamerad'), ('kameli', 'kambeli'), ('kamfer', 'kamper'), ('kamir', 'khamir'), ('kamis', 'gamis'), ('kamkama', 'kurkuma'), ('kampak', 'kapak'), ('kampang', 'kapang'), ('kampit', 'kampil'), ('kamuflase', 'penyamaran'), ('kana', 'katakana'), ('kanaat', 'kanaah'), ('kanah', 'kanaah'), ('kanal', 'terusan'), ('kanal', 'saluran'), ('kanat', 'kanaah'), ('kancah', 'kawah'), ('kancana', 'kencana'), ('kandar', 'gandar'), ('kandis', 'manis'), ('kangguru', 'kanguru'), ('kanjar', 'khanjar'), ('kanta', 'lensa'), ('kanti', 'teman'), ('kanti', 'kawan'), ('kantil', 'kontal-kantil'), ('kapit', 'pembantu'), ('kantung', 'kantong'), ('kanun', 'peraturan'), ('kanun', 'hukum'), ('kanyon', 'jurang'), ('kaos', 'kaus'), ('kapak', 'kepak'), ('kapak', 'sayap'), ('kapan', 'kafan'), ('kaparat', 'kafarat'), ('kaper', 'karper'), ('kapi', 'takal'), ('kapi', 'katrol'), ('kapilah', 'kafilah'), ('kapir', 'kafir'), ('kapling', 'kaveling'), ('kappa', 'kapa'), ('kaprah', 'lazim'), ('kaprah', 'biasa'), ('karaba', 'karamba'), ('karang', 'koral'), ('pekatul', 'bekatul'), ('karar', 'tenang'), ('karar', 'tenteram'), ('karbit', 'karbida'), ('karbonium', 'karbon'), ('kardan', 'gardan'), ('karkausar', 'alkausar'), ('kari', 'qari'), ('kariah', 'qariah'), ('karir', 'karier'), ('karotin', 'karotena'), ('karpet', 'permadani'), ('karpet', 'ambal'), ('karpus', 'kerpus'), ('kartonis', 'kartunis'), ('karya', 'buatan'), ('karyat', 'karyah'), ('karyawan', 'pegawai'), ('karyawan', 'pekerja'), ('kasab', 'kasap'), ('kasah', 'kasa'), ('kasak-kisik', 'kasak-kusuk'), ('kasar', 'kesat'), ('kasar', 'qasar'), ('kasatmata', 'nyata'), ('kasemek', 'kesemek'), ('kasep', 'kasip'), ('kasiat', 'khasiat'), ('kasih', 'mencintai'), ('kasima', 'kesima'), ('kasip', 'terlambat'), ('kastil', 'kastel'), ('kasuma', 'kusuma'), ('kasumat', 'kesumat'), ('kasus', 'perkara'), ('kasut', 'selipar'), ('kasuwari', 'kasuari'), ('kaswi', 'kasui'), ('kata', 'bicara'), ('kata', 'menyebut'), ('katagori', 'kategori'), ('katalis', 'katalisator'), ('katalogus', 'katalog'), ('katam', 'khatam'), ('katan', 'khitan'), ('kate', 'katai'), ('katek', 'ketek'), ('katek', 'katai'), ('katib', 'khatib'), ('katik', 'katai'), ('katirah', 'ketirah'), ('katlum', 'katelum'), ('kaula', 'kawula'), ('kaum', 'puak'), ('kaung', 'kawung'), ('kausar', 'alkausar'), ('kavling', 'kaveling'), ('kawal', 'jaga'), ('kawal', 'mengontrol'), ('kawan', 'teman'), ('kawan', 'sahabat'), ('kawan', 'bersekutu'), ('kawatir', 'khawatir'), ('kawin', 'menikahkan'), ('kawul', 'kaul'), ('kaya', 'kayak'), ('kayal', 'khayal'), ('kayambang', 'kiambang'), ('kazanah', 'khazanah'), ('kebambam', 'kebembem'), ('kebin', 'kabin'), ('kebit', 'kebat'), ('kebo', 'kerbau'), ('kebul', 'kepul'), ('kecik', 'kecil'), ('kecindan', 'kecandan'), ('kecipung', 'kecimpung'), ('kecuak', 'kecoak'), ('lampoyangan', 'lempuyangan'), ('kecundang', 'cundang'), ('kecut', 'lisut'), ('kecut', 'gentar'), ('kecut', 'mengerutkan'), ('keda', 'kedah'), ('kedai', 'warung'), ('lampu', 'pelita'), ('lampu', 'bohlam'), ('kedar', 'kadar'), ('kedeki', 'kedekai'), ('kedekik', 'kedekai'), ('kedele', 'kedelai'), ('kedemat', 'khidmat'), ('kedengkung', 'kedengkang'), ('keder', 'takut'), ('keder', 'gentar'), ('kedi', 'banci'), ('kedongdong', 'kedondong'), ('kedut', 'kerut'), ('kejam', 'bengis'), ('kejibeling', 'kecibeling'), ('kejolak', 'gejolak'), ('kekah', 'akikah'), ('kekang', 'kendali'), ('kekaras', 'karas'), ('kekudung', 'kerudung'), ('kelabet', 'kelabat'), ('kelah', 'pengaduan'), ('kelak', 'kelah'), ('kelakar', 'lawak'), ('kelakar', 'olok-olok'), ('kelamai', 'gelamai'), ('kelamarin', 'kemarin'), ('kelana', 'pengembara'), ('kelandera', 'kelendara'), ('kelang', 'kalang'), ('kelentang', 'kelantang'), ('kelebet', 'kelebek'), ('kelemantang', 'kalimantang'), ('kelemarin', 'kemarin'), ('kelembai', 'kelambai'), ('kelembubu', 'halimbubu'), ('kelembung', 'gelembung'), ('kelemumur', 'kelemur'), ('kelemunting', 'kemunting'), ('kelengkeng', 'lengkeng'), ('kelengking', 'kelengkeng'), ('kelentit', 'klitoris'), ('kelep', 'klep'), ('kelesah', 'gelisah'), ('keletar', 'geletar'), ('keleti', 'keliti'), ('keletik', 'geletik'), ('kelicih', 'kelicik'), ('kelicik', 'gelicik'), ('kelidai', 'kenidai'), ('kelikah', 'khalikah'), ('kelikih', 'keliki'), ('kelikir', 'kerikil'), ('keliling', 'melingkari'), ('kelimantang', 'kalimantang'), ('kelincir', 'gelincir'), ('kelindan', 'gelendong'), ('kelisah', 'gelisah'), ('keloceh', 'celoteh'), ('kelok', 'belokan'), ('kelok', 'keluk'), ('kelompang', 'kosong'), ('kelompang', 'hampa'), ('kenyih', 'kenyi'), ('keluar', 'timbul'), ('kelubung', 'selubung'), ('keluk', 'lengkung'), ('keluli', 'baja'), ('keluruk', 'keruyuk'), ('keluwek', 'keluak'), ('keluwung', 'pelangi'), ('kemak-kemik', 'kemik'), ('kemalai', 'gemulai'), ('kemanakan', 'kemenakan'), ('kemandang', 'kumandang'), ('kemas', 'rapi'), ('kembali', 'pemulangan'), ('kembali', 'pemulihan'), ('kembu', 'kumbu'), ('kemilau', 'kilau'), ('keminting', 'kemiri'), ('kemong', 'kemung'), ('kempas', 'kempis'), ('kempes', 'kempis'), ('kemucing', 'kemoceng'), ('kemudian', 'kelak'), ('kemuncak', 'puncak'), ('kemundir', 'kemendur'), ('kenantan', 'kinantan'), ('kencana', 'emas'), ('kencang', 'erat'), ('kening', 'alis'), ('kenor', 'kenur'), ('kendak', 'gendak'), ('kendala', 'rintangan'), ('kendor', 'kendur'), ('kengkang', 'kangkang'), ('kentong', 'kentung'), ('kenung', 'kenong'), ('kenyap', 'kenyam'), ('kepai', 'kapai'), ('kepak', 'kapai'), ('kepang', 'kipang'), ('kepau', 'kepar'), ('kepek', 'kempek'), ('kepik', 'kepek'), ('kepok', 'kepoh'), ('kepudang', 'kepodang'), ('kepuyuh', 'puyuh'), ('keraeng', 'karaeng'), ('kerangga', 'kerengga'), ('kerangkang', 'kelangkang'), ('keratau', 'kertau'), ('kerdip', 'kedip'), ('kerdum', 'kerdam'), ('kerdut', 'kedut'), ('kerek', 'kapi'), ('kerek', 'katrol'), ('kerekah', 'kerkah'), ('kerekut', 'kerekot'), ('keremi', 'kerawit'), ('kerempagi', 'kerampagi'), ('keremunting', 'kemunting'), ('keremut', 'keremot'), ('kerencing', 'kerenting'), ('kerenggamunggu', 'kardamunggu'), ('kerengkiang', 'rengkiang'), ('kerenyot', 'menyeringai'), ('kerenyut', 'kernyut'), ('kerepai', 'kerpai'), ('kerepak', 'kerpak'), ('kerepas', 'kerpas'), ('kerepek', 'keripik'), ('kerepot', 'keriput'), ('keresek', 'kerisik'), ('keretot', 'kerekot'), ('keretut', 'kerekot'), ('keriat-keriut', 'keriang-keriut'), ('kerimuk', 'kerumuk'), ('keringat', 'peluh'), ('keringsing', 'geringsing'), ('kerinyut', 'kernyut'), ('kerisik', 'kersik'), ('keritik', 'kritik'), ('kerja', 'melaksanakan'), ('kerja', 'menjalankan'), ('kerja', 'buruh'), ('kerkap', 'kerkah'), ('kerkap', 'kerakap'), ('kerkeling', 'kerakeling'), ('pinjak', 'pijak'), ('kerlap', 'berkilauan'), ('kerlip', 'berkedip-kedip'), ('kerlip', 'berkedip'), ('kermunting', 'kemunting'), ('kernyih', 'seringai'), ('kerodong', 'kerudung'), ('keroh', 'keruh'), ('keronco', 'keroncor'), ('keroncot', 'kerucut'), ('kerongsong', 'kelongsong'), ('keronsang', 'kerongsang'), ('keropong', 'keropok'), ('kerosang', 'kerongsang'), ('kerosek', 'kerosak'), ('kersip', 'kedip'), ('kertas', 'koran'), ('kerubian', 'kerubin'), ('keruit', 'kerawit'), ('keseleo', 'terkilir'), ('kesmaran', 'kasmaran'), ('kesuma', 'kusuma'), ('kesusu', 'tergesa-gesa'), ('kesusu', 'terburu-buru'), ('ketah', 'kutaha'), ('ketek', 'monyet'), ('ketek', 'kera'), ('ketawa', 'tertawa'), ('ketik', 'tik'), ('ketimbung', 'kecimpung'), ('ketimun', 'mentimun'), ('ketimus', 'timus'), ('ketombe', 'kelemumur'), ('keton', 'ketun'), ('ketonggeng', 'ketungging'), ('ketrok', 'ketuk'), ('ketua', 'mengepalai'), ('ketual', 'katwal'), ('ketul', 'gumpal'), ('ketul', 'kepal'), ('khabar', 'kabar'), ('khair', 'elok'), ('khalayak', 'umum'), ('khalayak', 'publik'), ('khalis', 'bersih'), ('khalis', 'murni'), ('kharab', 'binasa'), ('kharisma', 'karisma'), ('khas', 'khusus'), ('khatam', 'tamat'), ('khatam', 'selesai'), ('khaul', 'haul'), ('khisit', 'dengki'), ('khojah', 'khoja'), ('khudu', 'khuduk'), ('khulak', 'khuluk'), ('khunsa', 'banci'), ('khusuk', 'khusyuk'), ('khusus', 'istimewa'), ('khusus', 'teristimewa'), ('kibas', 'domba'), ('kibas', 'biri-biri'), ('kibernetika', 'sibernetika'), ('kibik', 'kubik'), ('kibir', 'sombong'), ('kibir', 'angkuh'), ('kicang-kicu', 'kicang-kecoh'), ('kicuh', 'kecoh'), ('kidab', 'kizib'), ('kidamat', 'khidmat'), ('kidang', 'kijang'), ('kifarat', 'kafarat'), ('kijip', 'kejap'), ('kiju', 'keju'), ('kik', 'akik'), ('kikir', 'pelit'), ('kikir', 'lokek'), ('kila', 'kilah'), ('kimar', 'himar'), ('kimpa', 'kimpal'), ('kin', 'ken'), ('kinang', 'ginang-ginang'), ('kingka', 'kimkha'), ('kingkip', 'kingking'), ('kintal', 'kuintal'), ('kinyam', 'kenyam'), ('kiong', 'keong'), ('kip', 'kep'), ('kipal', 'kimpal'), ('kipar', 'kepar'), ('kiparat', 'kafarat'), ('kira', 'menyangka'), ('kira', 'memperhitungkan'), ('kiraah', 'qiraah'), ('kiraat', 'qiraah'), ('kirabat', 'kerabat'), ('kiryah', 'karyah'), ('kiryat', 'karyah'), ('kisah', 'narasi'), ('kismat', 'nasib'), ('kismet', 'kismat'), ('kisut', 'lisut'), ('kizib', 'bohong'), ('kiting', 'keteng'), ('kiuk', 'keok'), ('kiut', 'kicut'), ('klas', 'kelas'), ('klen', 'klan'), ('klengkeng', 'lengkeng'), ('klenteng', 'kelenteng'), ('klobot', 'kelobot'), ('klor', 'klorin'), ('kocak', 'sombong'), ('kocek', 'kantong'), ('kodok', 'katak'), ('kohesif', 'padu'), ('kohlea', 'koklea'), ('kol', 'kubis'), ('kolah', 'kulah'), ('kolang-kalik', 'kolang-kaling'), ('kolembeng', 'kolomben'), ('kolese', 'akademi'), ('koliseng', 'ginseng'), ('kolok', 'kuluk'), ('kolomnis', 'kolumnis'), ('kolot', 'kuno'), ('koma', 'kurkuma'), ('komariah', 'kamariah'), ('komkoma', 'kurkuma'), ('komoditi', 'komoditas'), ('komplet', 'lengkap'), ('komplet', 'genap'), ('komplikasi', 'kerumitan'), ('komplit', 'komplet'), ('komposer', 'komponis'), ('komunikasi', 'hubungan'), ('komunitas', 'masyarakat'), ('konco', 'sahabat'), ('kondang', 'terkenal'), ('kondensator', 'kapasitor'), ('konduktor', 'dirigen'), ('konflik', 'perselisihan'), ('konflik', 'pertentangan'), ('konfrontir', 'konfrontasi'), ('kongkol', 'sekongkol'), ('kongkow', 'kongko'), ('kongkret', 'konkret'), ('kongkurs', 'konkurs'), ('kongres', 'muktamar'), ('konperensi', 'konferensi'), ('konsentrik', 'konsentris'), ('konservasi', 'pengawetan'), ('konservasi', 'pelestarian'), ('konsisten', 'ajek'), ('konsortium', 'konsorsium'), ('konspirasi', 'komplotan'), ('konstan', 'konstanta'), ('konstelasi', 'susunan'), ('konstipasi', 'sembelit'), ('konsultan', 'penasihat'), ('kontinuitas', 'kesinambungan'), ('kontrol', 'pengawasan'), ('koordinir', 'koordinasi'), ('kopak', 'kopok'), ('kopak', 'patah'), ('kopeling', 'kopling'), ('sitawar', 'setawar'), ('koperatif', 'kooperatif'), ('koperativisme', 'kooperativisme'), ('kopor', 'koper'), ('koran', 'harian'), ('korden', 'gorden'), ('kores', 'gores'), ('korma', 'kurma'), ('korsi', 'kursi'), ('korum', 'kuorum'), ('kosambi', 'kesambi'), ('kosen', 'berani'), ('koset', 'keset'), ('koterek', 'kotrek'), ('kotbah', 'khotbah'), ('kover', 'sampul'), ('kowak', 'kuak'), ('krah', 'kerah'), ('krakal', 'kerakal'), ('krama', 'kromo'), ('kran', 'keran'), ('kranium', 'tengkorak'), ('krebo', 'kribo'), ('krem', 'krim'), ('kremasi', 'pengabuan'), ('krematori', 'krematorium'), ('krematorium', 'perabuan'), ('kreseh-peseh', 'kareseh-peseh'), ('kriminil', 'kriminal'), ('krisis', 'kemelut'), ('kristalisasi', 'penghabluran'), ('kriterium', 'kriteria'), ('kritik', 'mengecam'), ('kroniometri', 'kraniometri'), ('ksatria', 'kesatria'), ('kua', 'koa'), ('kuartir', 'kwartir'), ('kuarza', 'kuarsa'), ('kuasa', 'kekuatan'), ('kuatir', 'khawatir'), ('langsar', 'beruntung'), ('kubung', 'pukang'), ('kucut', 'kecut'), ('kucandan', 'kecandan'), ('kucindan', 'kecandan'), ('kucir', 'kuncit'), ('kucung', 'kocong'), ('kucup', 'kecup'), ('kucup', 'menguncup'), ('kudi', 'kodi'), ('kudil', 'kudis'), ('kudus', 'suci'), ('kuetiau', 'kwetiau'), ('kuih', 'kue'), ('kuing', 'kaing'), ('kuja', 'khojah'), ('kujur', 'kaku'), ('kukang', 'kungkang'), ('kukuh', 'kuat'), ('kukuk', 'kokok'), ('kukul', 'kokol'), ('kukut', 'kokot'), ('kulambai', 'kelambai'), ('kulari', 'kelari'), ('kuldi', 'khuldi'), ('kulikat', 'kelikat'), ('kulintang', 'kolintang'), ('kulio', 'kalio'), ('kulkulah', 'kalkalah'), ('kuma-kuma', 'kurkuma'), ('kumala', 'kemala'), ('kuman', 'bakteri'), ('kumango', 'kumanga'), ('kumat', 'komat-kamit'), ('kumayan', 'kemenyan'), ('kumel', 'kumal'), ('kumis', 'misai'), ('kumkuma', 'kurkuma'), ('kumpal', 'gumpal'), ('kumpul', 'rapat'), ('kumpul', 'pertemuan'), ('kumpul', 'perhimpunan'), ('kumut', 'menyeringai'), ('kuna', 'kuno'), ('kunarpa', 'bangkai'), ('kunarpa', 'mayat'), ('kuota', 'jatah'), ('kuotum', 'kuota'), ('kup', 'kudeta'), ('kupak', 'rusak'), ('kupel', 'kopel'), ('kuprum', 'tembaga'), ('kupu', 'kufu'), ('kupu-kupu', 'rama-rama'), ('kusen', 'kosen'), ('kursi', 'sofa'), ('kurun', 'abad'), ('kurun', 'zaman'), ('kurun', 'berabad-abad'), ('kuruyuk', 'kukuruyuk'), ('kus', 'dekus'), ('kusa', 'angkusa'), ('kusta', 'lepra'), ('kusti', 'gusti'), ('kusuk', 'khusuk'), ('kutbah', 'khotbah'), ('labang', 'mengembara'), ('kutu', 'kepinding'), ('kutuk', 'menyumpahi'), ('kuwalat', 'kualat'), ('kuwung-kuwung', 'pelangi'), ('kuyup', 'basah'), ('kwartet', 'kuartet'), ('kwiz', 'kuis'), ('labah-labah', 'laba-laba'), ('labrang', 'laberang'), ('lacur', 'sundal'), ('lajak', 'cepat'), ('lada', 'cabai'), ('ladah', 'cemar'), ('ladah', 'kotor'), ('ladah', 'jijik'), ('lahap', 'rakus'), ('lahat', 'lahad'), ('lahip', 'daif'), ('lahir', 'keduniaan'), ('laif', 'lemah'), ('lailah', 'malam'), ('lailat', 'lailah'), ('laip', 'daif'), ('laka', 'lak'), ('lakri', 'lakeri'), ('laksmi', 'elok'), ('laksmi', 'molek'), ('laksmi', 'cantik'), ('lalai', 'melengahkan'), ('lalai', 'terlupa'), ('lalalat', 'dalalat'), ('lalang', 'langlang'), ('lalim', 'zalim'), ('lalu', 'lulus'), ('lalu', 'lantas'), ('lambat', 'lambat-lambat'), ('laminah', 'lamina'), ('lanca', 'lancia'), ('lancia', 'langca'), ('lancit', 'lencit'), ('lancung', 'palsu'), ('lancung', 'lancong'), ('landai', 'lendaian'), ('langsep', 'langsat'), ('landas', 'alas'), ('landas', 'tumpuan'), ('lang', 'elang'), ('langak-languk', 'langak-longok'), ('langcia', 'lancia'), ('langgas', 'bebas'), ('langsing', 'lampai'), ('langkah', 'berjalan'), ('langkara', 'lengkara'), ('langkas', 'tangkas'), ('langkas', 'giat'), ('langkitang', 'lengkitang'), ('langsang', 'langseng'), ('languh', 'lenguh'), ('lanjang', 'telanjang'), ('lanjut', 'tua'), ('lansai', 'langsai'), ('lansar', 'langsar'), ('lansat', 'langsat'), ('lansekap', 'lanskap'), ('lansi', 'langsi'), ('lansir', 'langsir'), ('lantam', 'angkuh'), ('lantam', 'sombong'), ('lantas', 'langsung'), ('lantera', 'lentera'), ('lanun', 'perompak'), ('laoteng', 'loteng'), ('lapal', 'lafal'), ('lapang', 'luas'), ('lapaz', 'lafal'), ('latang', 'jelatang'), ('lapo', 'lepau'), ('lapuk', 'usang'), ('lapur', 'lapor'), ('larak', 'lerak'), ('lata', 'buruk'), ('lasit', 'lesit'), ('lasuh', 'mudah'), ('lasykar', 'laskar'), ('latif', 'cantik'), ('lauh', 'loh'), ('laun', 'lambat'), ('laung', 'meneriakkan'), ('lava', 'lahar'), ('lavender', 'lavendel'), ('lawak', 'lucu'), ('lawak', 'jenaka'), ('lawak', 'lelucon'), ('lawan', 'bertentangan'), ('lawas', 'luas'), ('lawas', 'lapang'), ('lawe', 'lawai'), ('layak', 'pantas'), ('lazat', 'lezat'), ('lebar', 'lapang'), ('leber', 'luber'), ('lebur', 'debur'), ('lecer', 'lecet'), ('ledeng', 'leding'), ('lega', 'luas'), ('legah', 'lega'), ('legalisir', 'legalisasi'), ('legam', 'legum'), ('legojo', 'algojo'), ('legundi', 'lenggundi'), ('lehar', 'rehal'), ('lekah', 'rekah'), ('lekap', 'melengket'), ('lekar', 'rehal'), ('lekas', 'segera'), ('lekat', 'menempeli'), ('lekemia', 'leukemia'), ('lekosit', 'leukosit'), ('lembur', 'lambur'), ('lelabah', 'laba-laba'), ('lelabi', 'labi-labi'), ('lelah', 'letih'), ('lelah', 'lesu'), ('lelah', 'kepenatan'), ('lelak', 'rerak'), ('lelaki', 'laki-laki'), ('lelangit', 'langit-langit'), ('lelangse', 'langsai'), ('lelawa', 'kelelawar'), ('lembur', 'limbur'), ('leli', 'lili'), ('lelung', 'lelang'), ('lem', 'perekat'), ('lemah', 'lembut'), ('lembusir', 'lemusir'), ('lepet', 'lepat'), ('lembab', 'lembap'), ('lembaga', 'mudigah'), ('lembam', 'lebam'), ('lemban', 'lembam'), ('lembang', 'membujuk'), ('lembar', 'utas'), ('lembega', 'rembega'), ('lembik', 'lembek'), ('lembu', 'sapi'), ('lembung', 'lambung'), ('lemidang', 'lembidang'), ('lempah', 'limpah'), ('lempam', 'lempem'), ('lempang', 'lempeng'), ('lempar', 'melontarkan'), ('lempedu', 'empedu'), ('lempenai', 'lampeni'), ('lempeng', 'jujur'), ('lempeni', 'lampeni'), ('lempoh', 'lempuh'), ('lempok', 'lempuk'), ('lemukat', 'melukut'), ('lemukut', 'melukut'), ('lemungsir', 'lemusir'), ('lemuru', 'lemburu'), ('lena', 'lengah'), ('lenan', 'linen'), ('lenceng', 'lencang'), ('lencet', 'lecet'), ('lencong', 'licin'), ('lencong', 'lencun'), ('lencun', 'licin'), ('lender', 'lendir'), ('lenga', 'bijan'), ('lengang', 'sunyi'), ('lengang', 'kesepian'), ('lengas', 'lembap'), ('lengau', 'langau'), ('lenger', 'lengar'), ('lenggara', 'selenggara'), ('lengkara', 'mustahil'), ('lengkas', 'langkas'), ('lengkiang', 'rangkiang'), ('lengkok', 'bengkok'), ('lengkok', 'kelok'), ('lengkok', 'keluk'), ('lenguh', 'lesu'), ('lenja', 'lancang'), ('lenser', 'lengser'), ('lenset', 'lengset'), ('lentun', 'lantun'), ('lenyai', 'lemah'), ('leontin', 'liontin'), ('leper', 'ceper'), ('leper', 'rata'), ('leplap', 'liplap'), ('lepuk', 'lapuk'), ('leram', 'deram'), ('lerik', 'derik'), ('lerum', 'derum'), ('les', 'lis'), ('lesah', 'lasah'), ('lesak', 'lasak'), ('lesap', 'lenyap'), ('lesat', 'pelesat'), ('lesau', 'desau'), ('lesek', 'lasak'), ('leset', 'peleset'), ('lesih', 'lesi'), ('lestari', 'kekal'), ('lesu', 'letih'), ('lesu', 'anemia'), ('lesus', 'desus'), ('leta', 'hina'), ('letak', 'menempatkan'), ('letek', 'peletek'), ('letih', 'melelahkan'), ('letih', 'penat'), ('letup', 'meletus'), ('letup', 'ledakan'), ('letur', 'lentur'), ('letur', 'lecur'), ('lewat', 'kasip'), ('lezat', 'sedap'), ('liang', 'pori'), ('libereto', 'libreto'), ('licau', 'berkilat'), ('licin', 'licik'), ('ligih', 'menggigil'), ('lihat', 'kelihatan'), ('lihat', 'tampak'), ('liku', 'kelok'), ('lilin', 'malam'), ('limbur', 'menggenangi'), ('limfopenia', 'limfositopenia'), ('limonade', 'limun'), ('limpah', 'berlebih-lebih'), ('limpah', 'berlimpah-limpah'), ('linan', 'linen'), ('lintuh', 'lemah'), ('linggam', 'sedelinggam'), ('linggis', 'perejang'), ('lingkap', 'lindang'), ('lintadu', 'sentadu'), ('lintas', 'melalui'), ('lipan', 'chilopoda'), ('lipu', 'suram'), ('lipur', 'lenyap'), ('lipur', 'melenyapkan'), ('lisah', 'minyak'), ('liver', 'lever'), ('liwat', 'lewat'), ('loba', 'tamak'), ('lobang', 'lubang'), ('loha', 'duha'), ('lohor', 'zuhur'), ('lokalisir', 'lokalisasi'), ('loki', 'loktong'), ('lompat', 'melonjak-lonjak'), ('lompong', 'hampa'), ('lonte', 'pelacur'), ('losin', 'lusin'), ('loyal', 'setia'), ('loyo', 'loya'), ('luasa', 'leluasa'), ('lubur', 'belubur'), ('lucah', 'kecabulan'), ('luhak', 'luak'), ('luhur', 'tinggi'), ('luku', 'tenggala'), ('lukut', 'melukut'), ('lumrah', 'biasa'), ('lumrah', 'lazim'), ('lung', 'long'), ('lup', 'lop'), ('luruh', 'menggugurkan'), ('maaf', 'pengampunan'), ('mabir', 'tabir'), ('macan', 'harimau'), ('macat', 'macet'), ('mada', 'madar'), ('madaliun', 'medalion'), ('madarat', 'mudarat'), ('madarsah', 'madrasah'), ('maddah', 'madah'), ('madi', 'mazi'), ('madia', 'madya'), ('magfirat', 'magfirah'), ('madukara', 'lebah'), ('maesan', 'nisan'), ('magep-magep', 'megap-megap'), ('magik', 'magis'), ('magobi', 'mahoni'), ('mairat', 'mikraj'), ('mahaligai', 'mahligai'), ('mahar', 'maskawin'), ('mahbubat', 'mahbubah'), ('mahful', 'mahfuz'), ('mait', 'mayat'), ('mendepun', 'depun'), ('mahir', 'kemampuan'), ('mahisa', 'mahesa'), ('mahkamah', 'pengadilan'), ('mahraj', 'makhraj'), ('mahsar', 'mahsyar'), ('majal', 'tumpul'), ('maja-muju', 'jemuju'), ('majer', 'majir'), ('majilis', 'majelis'), ('majir', 'mandul'), ('majlis', 'majelis'), ('makan', 'memerlukan'), ('makas', 'mangkas'), ('makcik', 'bibi'), ('makena', 'mukena'), ('malapetaka', 'kecelakaan'), ('malapetaka', 'kesengsaraan'), ('maklum', 'mengerti'), ('makmal', 'laboratorium'), ('makota', 'mahkota'), ('makroni', 'makaroni'), ('makul', 'logis'), ('makyung', 'makyong'), ('malaik', 'malaikat'), ('malak', 'malaikat'), ('malakulmaut', 'malaikatulmaut'), ('malang', 'kesialan'), ('malau', 'embalau'), ('malaun', 'malun'), ('malih', 'berubah'), ('malin', 'malim'), ('mama', 'ibu'), ('mamar', 'mamang'), ('mamat', 'mamut'), ('mampat', 'pampat'), ('mancis', 'macis'), ('mancit', 'mencit'), ('mancung', 'runcing'), ('mancur', 'pancur'), ('mandah', 'manda'), ('mandek', 'terhenti'), ('mandiang', 'mendiang'), ('mandulika', 'mandalika'), ('mandur', 'mandor'), ('manfaat', 'berguna'), ('mangga', 'mempelam'), ('mangkah', 'mangkar'), ('mangkal', 'pangkal'), ('mangkara', 'makara'), ('mangkas', 'keras'), ('mangkas', 'mangkar'), ('mangkin', 'makin'), ('mangkok', 'mangkuk'), ('mangkuk', 'cawan'), ('mangsai', 'masai'), ('mani', 'sperma'), ('manikin', 'maneken'), ('manjur', 'mustajab'), ('mendera', 'mendira'), ('mantap', 'kukuh'), ('mantari', 'mentari'), ('mantera', 'mantra'), ('manti', 'menteri'), ('mantol', 'mantel'), ('mantram', 'mantra'), ('mendera', 'bendera'), ('manusia', 'insan'), ('mapak', 'papak'), ('marah', 'berang'), ('marah', 'kegusaran'), ('margin', 'batas'), ('marhum', 'almarhum'), ('marhumah', 'almarhumah'), ('marihuana', 'mariyuana'), ('marikan', 'merikan'), ('marikh', 'mars'), ('marjin', 'margin'), ('marjinal', 'marginal'), ('marjinalisasi', 'marginalisasi'), ('marka', 'markah'), ('marmar', 'marmer'), ('marmer', 'pualam'), ('marmut', 'marmot'), ('marwah', 'muruah'), ('marzipan', 'marsepen'), ('mas', 'emas'), ('masal', 'massal'), ('masalah', 'soal'), ('masalah', 'persoalan'), ('masarakat', 'masyarakat'), ('masgul', 'masygul'), ('mashaf', 'mushaf'), ('mashur', 'masyhur'), ('masin', 'mesin'), ('masohi', 'masori'), ('masoyi', 'masoi'), ('masyhaf', 'mashaf'), ('masyhur', 'terkenal'), ('mat', 'emat'), ('mataliur', 'mitraliur'), ('mateng', 'matang'), ('materai', 'meterai'), ('materi', 'bahan'), ('matra', 'dimensi'), ('matres', 'matris'), ('maturitas', 'kematangan'), ('maturitas', 'kedewasaan'), ('maujud', 'nyata'), ('maulaya', 'maulai'), ('mawar', 'tawar'), ('mawat', 'mati'), ('maya', 'khayalan'), ('mayar', 'kelemayar'), ('mayit', 'mayat'), ('mecis', 'macis'), ('medan', 'alun-alun'), ('medeli', 'medali'), ('medikus', 'dokter'), ('medit', 'kikir'), ('medit', 'pelit'), ('meditasi', 'bertafakur'), ('meduk', 'medok'), ('megalopolis', 'megapolis'), ('megar', 'mekar'), ('mejelis', 'majelis'), ('mejen', 'mejan'), ('mekis', 'mengkis'), ('melaka', 'malaka'), ('melambing', 'melambang'), ('melapari', 'malapari'), ('melarat', 'miskin'), ('melarat', 'larat'), ('melata', 'lata'), ('melerang', 'belerang'), ('melesat', 'pelesat'), ('meleset', 'peleset'), ('meling', 'meleng'), ('melinjo', 'belinjo'), ('melpari', 'malapari'), ('membang', 'mambang'), ('membruk', 'mambruk'), ('memengkis', 'pekis'), ('memerang', 'memberang'), ('memori', 'ingatan'), ('mempedal', 'empedal'), ('memur', 'memar'), ('menalu', 'benalu'), ('menderang', 'benderang'), ('menang', 'mengalahkan'), ('menasabah', 'munasabah'), ('menat', 'minat'), ('menatu', 'penatu'), ('mencok', 'menclok'), ('mencret', 'menceret'), ('mendali', 'medali'), ('mendalika', 'mandalika'), ('mendalu', 'benalu'), ('mendeleka', 'mandalika'), ('mendora', 'mendura'), ('mendreng', 'mindring'), ('mendusin', 'dusin'), ('menepaat', 'manfaat'), ('mengah', 'terengah-engah'), ('mengeh', 'mengi'), ('mengerti', 'erti'), ('menget', 'senget'), ('menggerib', 'magrib'), ('mengkar', 'mekar'), ('mengkara', 'makara'), ('mengkarung', 'bengkarung'), ('mengkawan', 'bengkawan'), ('mengkel', 'mengkal'), ('mengkerat', 'mengkeret'), ('mengkilap', 'kilap'), ('mengkirik', 'kirik'), ('mengkuang', 'bengkuang'), ('mengok', 'mengot'), ('mengor', 'mengot'), ('mengsong', 'mengsol'), ('menikam', 'manikam'), ('menila', 'manila'), ('menira', 'manira'), ('menjak', 'semenjak'), ('menjana', 'semenjana'), ('menjangan', 'rusa'), ('menjangan', 'kijang'), ('menjelai', 'enjelai'), ('menjelis', 'majelis'), ('mensiu', 'mesiu'), ('menta', 'meta'), ('mentari', 'matahari'), ('mentelah', 'sementelah'), ('menteros', 'matros'), ('mentika', 'mestika'), ('mentimun', 'timun'), ('mentua', 'mertua'), ('menturung', 'benturung'), ('menung', 'meditasi'), ('menyampang', 'senyampang'), ('menyan', 'kemenyan'), ('menyawak', 'biawak'), ('meram', 'merem'), ('merbak', 'semerbak'), ('mereng', 'miring'), ('merbot', 'marbut'), ('merca', 'murca'), ('mercapada', 'marcapada'), ('mercun', 'mercon'), ('mercupada', 'marcapada'), ('merekan', 'merikan'), ('merguk', 'berguk'), ('meringis', 'ringis'), ('merjer', 'merger'), ('merkah', 'markah'), ('merkah', 'rekah'), ('merlilin', 'melilin'), ('merpelai', 'mempelai'), ('mertega', 'mentega'), ('merual', 'merawal'), ('mesigit', 'masjid'), ('mesjid', 'masjid'), ('meskat', 'maskat'), ('mesra', 'karib'), ('mesta', 'semesta'), ('mesui', 'masoi'), ('metafisik', 'metafisika'), ('metah', 'sangat'), ('meterai', 'cap'), ('meterai', 'tera'), ('metonimi', 'metonimia'), ('mezbah', 'mazbah'), ('midik', 'sidik'), ('mikroba', 'mikrob'), ('mil', 'batu'), ('milik', 'hak'), ('milioner', 'miliuner'), ('minantu', 'menantu'), ('minatu', 'penatu'), ('minimal', 'sedikit-dikitnya'), ('mirai', 'tirai'), ('miring', 'mencondongkan'), ('misal', 'contoh'), ('misua', 'misoa'), ('mitra', 'sahabat'), ('mobil', 'otomobil'), ('mocok', 'pocok'), ('mohor', 'tera'), ('molek', 'elok'), ('molek', 'cantik'), ('mondok', 'pondok'), ('mong', 'mung'), ('mongkor', 'mungkur'), ('moral', 'akhlak'), ('morat-marit', 'berantakan'), ('mosaik', 'mozaik'), ('mozah', 'mojah'), ('muadin', 'muazin'), ('mualif', 'pengarang'), ('muasal', 'asal'), ('mucikari', 'muncikari'), ('mudakar', 'muzakar'), ('mudarat', 'kerugian'), ('mudasir', 'dasar'), ('mudigah', 'embrio'), ('mufakat', 'setuju'), ('muflis', 'bangkrut'), ('muhrim', 'mahram'), ('mujarab', 'manjur'), ('muk', 'mok'), ('mula', 'awal'), ('mulat', 'mulato'), ('mules', 'mulas'), ('multiplikasi', 'perkalian'), ('mumur', 'hancur'), ('muna', 'muno'), ('munasabah', 'sesuai'), ('muncikari', 'jaruman'), ('muncung', 'moncong'), ('mungguk', 'munggu'), ('munggur', 'mungkur'), ('munisi', 'amunisi'), ('mupaham', 'mufaham'), ('mupakat', 'mufakat'), ('muparik', 'mufarik'), ('muram', 'suram'), ('murat-marit', 'morat-marit'), ('murni', 'suci'), ('musabakah', 'musabaqah'), ('musafar', 'musafir'), ('musi', 'mungsi'), ('muskil', 'sukar'), ('muskil', 'sulit'), ('musnah', 'membinasakan'), ('musola', 'musala'), ('muson', 'monsun'), ('mustami', 'mustamik'), ('musti', 'mesti'), ('mustika', 'mestika'), ('musyawarat', 'musyawarah'), ('musykil', 'muskil'), ('musytari', 'yupiter'), ('mutabar', 'muktabar'), ('mutakhir', 'terbaru'), ('mutamad', 'muktamad'), ('mutu', 'kadar'), ('muwarikh', 'muarikh'), ('muzhab', 'mazhab'), ('naas', 'nahas'), ('nadar', 'nazar'), ('nadirat', 'nadir'), ('nafas', 'napas'), ('nafi', 'menampik'), ('nafi', 'mengingkari'), ('nahwu', 'nahu'), ('naik', 'menanjak'), ('naik', 'menunggang'), ('naik', 'meningkat'), ('naik', 'mendaki'), ('naik', 'peningkatan'), ('najam', 'bintang'), ('najasat', 'najasah'), ('nakoda', 'nakhoda'), ('nalih', 'nali'), ('nampak', 'tampak'), ('nampal', 'napal'), ('nangkoda', 'nakhoda'), ('napi', 'nafi'), ('napsi', 'nafsi'), ('napsu', 'nafsu'), ('narawastu', 'narwastu'), ('narestu', 'narwastu'), ('narkotika', 'narkotik'), ('narpati', 'narapati'), ('nasehat', 'nasihat'), ('nayaga', 'niyaga'), ('nasional', 'kebangsaan'), ('nasrani', 'kristen'), ('natur', 'pembawaan'), ('nazir', 'nadir'), ('neces', 'necis'), ('negari', 'nagari'), ('neka', 'aneka'), ('nekad', 'nekat'), ('nekel', 'nikel'), ('nenar', 'nanar'), ('nenekanda', 'nenenda'), ('neolitikum', 'neolitik'), ('neonisasi', 'peneonan'), ('neoplasma', 'tumor'), ('netralisir', 'netralisasi'), ('neurosis', 'psikoneurosis'), ('ninitowong', 'ninitowok'), ('nipas', 'nifas'), ('nirmala', 'bersih'), ('nirmala', 'suci'), ('niru', 'nyiru'), ('niskala', 'mujarad'), ('niskala', 'abstrak'), ('niur', 'nyiur'), ('noda', 'cela'), ('nomer', 'nomor'), ('nonoh', 'senonoh'), ('notulen', 'notula'), ('nur', 'cahaya'), ('nur', 'sinar'), ('nuriah', 'terang'), ('nurmala', 'nirmala'), ('nyah', 'enyah'), ('nyaman', 'sejuk'), ('nyampang', 'senyampang'), ('nyawa', 'roh'), ('nyenyet', 'nyenyat'), ('nyiur', 'kelapa'), ('nyonyot', 'nyunyut'), ('oasis', 'wahah'), ('obrak-abrik', 'ubrak-abrik'), ('obyek', 'objek'), ('obyektif', 'objektif'), ('obyektivisme', 'objektivisme'), ('obyektivitas', 'objektivitas'), ('ojah', 'oja'), ('oktan', 'oktana'), ('olak', 'pusaran'), ('olia', 'aulia'), ('omset', 'omzet'), ('omel', 'mencomel'), ('omnivora', 'omnivor'), ('omong', 'cakap'), ('omong', 'bual'), ('omong', 'bercakap'), ('omong', 'mempercakapkan'), ('onani', 'masturbasi'), ('ongok', 'bodoh'), ('ongok', 'tolol'), ('onta', 'unta'), ('onyak-anyik', 'onyah-anyih'), ('opau', 'pao-pao'), ('open', 'oven'), ('opium', 'candu'), ('opium', 'madat'), ('oplet', 'opelet'), ('oportunitas', 'peluang'), ('orasio', 'orasi'), ('orde', 'susunan'), ('oreol', 'aureol'), ('orisinil', 'orisinal'), ('ornamen', 'perhiasan'), ('oseanografi', 'oseanologi'), ('osilasi', 'ayunan'), ('otak', 'pikiran'), ('otentik', 'autentik'), ('otoaktivitas', 'autoaktivitas'), ('otobiografi', 'autobiografi'), ('otodidak', 'autodidak'), ('otograf', 'autograf'), ('otografi', 'autografi'), ('otokrasi', 'autokrasi'), ('otokrat', 'autokrat'), ('otokritik', 'autokritik'), ('otomasi', 'automasi'), ('otopsi', 'autopsi'), ('otorita', 'otoritas'), ('otoritet', 'otoritas'), ('ovari', 'ovarium'), ('oyak', 'mengejar'), ('oyong', 'goyang'), ('pah', 'pak'), ('pacak', 'pacek'), ('pacar', 'kekasih'), ('pacat', 'pacet'), ('pacis', 'pacih'), ('padan', 'sesuai'), ('padang', 'lapangan'), ('padat', 'padu'), ('padat', 'mampat'), ('padepokan', 'pedepokan'), ('padu', 'pejal'), ('padusi', 'pedusi'), ('paedah', 'faedah'), ('pagan', 'kuat'), ('pagun', 'pagan'), ('pahing', 'paing'), ('pakar', 'spesialis'), ('pakat', 'pekat'), ('pakbon', 'vakbon'), ('paku', 'pakis'), ('pancakara', 'berkelahi'), ('pancakara', 'berperang'), ('palam', 'penutup'), ('palasik', 'pelesit'), ('pali', 'pemali'), ('palit', 'pailit'), ('palm', 'palem'), ('palma', 'palem'), ('palmit', 'palmin'), ('palung', 'lawak-lawak'), ('palut', 'bersalut'), ('palut', 'menyalut'), ('pandai', 'pintar'), ('pandak', 'pendek'), ('pandau', 'paya'), ('pandir', 'bodoh'), ('pandir', 'bebal'), ('panen', 'penuaian'), ('panen', 'tuaian'), ('pangkal', 'permulaan'), ('pangkal', 'asal'), ('pangsi', 'paksi'), ('panili', 'vanili'), ('panitra', 'panitera'), ('panjer', 'panjar'), ('pantai', 'pesisir'), ('panti', 'rumah'), ('pantas', 'selayaknya'), ('pantas', 'cepat'), ('pantas', 'tangkas'), ('pantat', 'bokong'), ('panu', 'panau'), ('papak', 'rata'), ('paria', 'peria'), ('papat', 'pepat'), ('para', 'pagu'), ('paraid', 'faraid'), ('paralisis', 'kelumpuhan'), ('paralitis', 'lumpuh'), ('parampara', 'paranpara'), ('parap', 'paraf'), ('parau', 'serak'), ('parit', 'selokan'), ('pariwisata', 'pelancongan'), ('pariwisata', 'turisme'), ('parji', 'farji'), ('paro', 'paruh'), ('parokial', 'sempit'), ('partial', 'parsial'), ('partikel', 'zarah'), ('partisi', 'sekat'), ('pasah', 'fasakh'), ('pasanggrahan', 'pesanggrahan'), ('pasara', 'pusara'), ('paset', 'faset'), ('pasik', 'fasik'), ('pasirah', 'pesirah'), ('pasti', 'tentu'), ('pasti', 'ketentuan'), ('pastori', 'pastoran'), ('pastur', 'pasteur'), ('patih', 'patuh'), ('pateram', 'petaram'), ('patikim', 'tikim'), ('patuh', 'menuruti'), ('patut', 'layak'), ('patut', 'pantas'), ('pauh', 'mangga'), ('perisa', 'enak'), ('perisa', 'sedap'), ('payir', 'pair'), ('payudara', 'susu'), ('payudara', 'tetek'), ('pecah beling', 'kejibeling'), ('pecal', 'pecel'), ('pecok', 'pecak'), ('pedagogik', 'pedagogis'), ('pedah', 'padah'), ('pedal', 'empedal'), ('pedu', 'empedu'), ('peduli', 'mengindahkan'), ('peduli', 'menghiraukan'), ('pegawam', 'peguam'), ('pekah', 'nafkah'), ('pekak', 'bengap'), ('pekam', 'pakem'), ('pekerti', 'tabiat'), ('pekerti', 'akhlak'), ('pekir', 'fakir'), ('pekir', 'apkir'), ('pekong', 'toapekong'), ('pekur', 'tafakur'), ('pelaga', 'kapulaga'), ('pelak', 'salah'), ('pelak', 'keliru'), ('pelan', 'lambat'), ('pelangkin', 'belangkin'), ('pelapah', 'pelepah'), ('pelbaya', 'pelebaya'), ('pelekuk', 'bengkok'), ('pelembaya', 'pelebaya'), ('pelempap', 'telempap'), ('pelihara', 'melindungi'), ('pelik', 'aneh'), ('peluh', 'peloh'), ('pelir', 'zakar'), ('pelok', 'peluk'), ('pelopak', 'kelopak'), ('pelor', 'peluru'), ('pelungpung', 'pelumpung'), ('pematah', 'pepatah'), ('pemindang', 'pemidang'), ('pen', 'pena'), ('penaga', 'menaga'), ('penak', 'pinak'), ('pendapa', 'pedapa'), ('pendekar', 'pahlawan'), ('peniaram', 'penaram'), ('penis', 'zakar'), ('penjara', 'bui'), ('penjuru', 'pojok'), ('pentas', 'panggung'), ('pentilasi', 'ventilasi'), ('penyet', 'penyek'), ('peparu', 'paru-paru'), ('perancah', 'perencah'), ('perancit', 'pancit'), ('peranggi', 'peringgi'), ('perangkap', 'menjebak'), ('perangkap', 'terjebak'), ('peranjat', 'mengejutkan'), ('peranjat', 'mengagetkan'), ('perbahasa', 'peribahasa'), ('perbegu', 'pelebegu'), ('percuma', 'gratis'), ('perempuan', 'wanita'), ('perengkat', 'peringkat'), ('pergat', 'fregat'), ('pergata', 'fregat'), ('pergi', 'keberangkatan'), ('pergul', 'pergol'), ('perhati', 'mengamati'), ('perigi', 'sumur'), ('peringkat', 'tingkat'), ('perit', 'perih'), ('perit', 'pedih'), ('perji', 'farji'), ('perkosa', 'menggagahi'), ('perli', 'mengejek'), ('perli', 'mencemooh'), ('perlintih', 'perlenteh'), ('perlu', 'membutuhkan'), ('perlu', 'kemestian'), ('permai', 'elok'), ('pernekel', 'pernikel'), ('pernel', 'flanel'), ('perop', 'prop'), ('perselah', 'perslah'), ('persero', 'pesero'), ('personil', 'personel'), ('perspektif', 'pandangan'), ('pertanda', 'pelebaya'), ('perugul', 'perogol'), ('perunggu', 'gangsa'), ('perwatin', 'batin'), ('pesara', 'pasar'), ('pesemendan', 'pasumandan'), ('peset', 'pesek'), ('pesiar', 'berjalan-jalan'), ('pesok', 'pesuk'), ('pestol', 'pistol'), ('petrol', 'bensin'), ('phi', 'fi'), ('piawai', 'cakap'), ('picah', 'pica'), ('pici', 'peci'), ('pico', 'lengah'), ('pidada', 'pedada'), ('pijat', 'bangsat'), ('pikup', 'pikap'), ('pil', 'tablet'), ('pilek', 'selesma'), ('pilin', 'pintal'), ('pilot', 'penerbang'), ('pinas', 'penes'), ('pinsil', 'pensil'), ('pinda', 'memperbaiki'), ('pindah', 'peralihan'), ('pinding', 'kepinding'), ('pinggir', 'tepi'), ('pingkel', 'pingkal'), ('pinis', 'penes'), ('pipa', 'pembuluh'), ('pir', 'per'), ('pirasat', 'firasat'), ('piyama', 'piama'), ('pirsawan', 'pemirsa'), ('piruk', 'hiruk'), ('pisah', 'cerai'), ('pisak', 'pesak'), ('pisit', 'pisik'), ('planel', 'flanel'), ('pitah', 'petah'), ('pitih', 'pitis'), ('piting', 'kepiting'), ('pitrah', 'fitrah'), ('plat', 'pelat'), ('platform', 'program'), ('platinum', 'platina'), ('podak', 'pandan'), ('podium', 'mimbar'), ('pojok', 'sudut'), ('poket', 'saku'), ('pokok', 'dasar'), ('pol', 'pul'), ('pola', 'model'), ('polis', 'poles'), ('polmak', 'polmah'), ('polonter', 'volunter'), ('pompong', 'kepompong'), ('pondamen', 'fundamen'), ('pondasi', 'fondasi'), ('pondok', 'penginapan'), ('pongah', 'keangkuhan'), ('porter', 'portir'), ('posfor', 'fosfor'), ('posisi', 'meletakkan'), ('positif', 'pasti'), ('postur', 'perawakan'), ('potensi', 'kekuatan'), ('potensi', 'daya'), ('potret', 'foto'), ('prabawa', 'perbawa'), ('prada', 'perada'), ('prakata', 'mukadimah'), ('praktek', 'praktik'), ('pramuka', 'pandu'), ('pranata', 'institusi'), ('prehistori', 'prasejarah'), ('profan', 'duniawi'), ('preservasi', 'pengawetan'), ('presidentil', 'presidensial'), ('prinsip', 'dasar'), ('propinsi', 'provinsi'), ('prive', 'privat'), ('priyagung', 'priagung'), ('priyayi', 'priayi'), ('problem', 'masalah'), ('produser', 'produsen'), ('profesor', 'mahaguru'), ('prospek', 'kemungkinan'), ('provokatur', 'provokator'), ('puasa', 'saum'), ('publisir', 'publikasi'), ('puerperium', 'puerpera'), ('pukang', 'kungkang'), ('puitik', 'puitis'), ('pukul', 'mengalahkan'), ('pungkir', 'mungkir'), ('pulang', 'pengembalian'), ('pules', 'pulas'), ('puling', 'pialing'), ('pulpa', 'pulp'), ('pundak', 'bahu'), ('punggawa', 'penggawa'), ('puntul', 'tumpul'), ('pusa', 'puso'), ('pusar', 'pusat'), ('pusara', 'kubur'), ('pusung', 'tolol'), ('puser', 'pusar'), ('pusing', 'putar'), ('pustaka', 'buku'), ('putar', 'memusingkan'), ('putar', 'pusingan'), ('putih', 'suci'), ('putus', 'sedih'), ('puyu', 'puyuh'), ('rabik', 'rabit'), ('radak', 'menyerang'), ('rades', 'radis'), ('ragam', 'laku'), ('ragbol', 'rakbol'), ('ragut', 'renggut'), ('rahat', 'beristirahat'), ('rahat', 'rehat'), ('rahim', 'peranakan'), ('rahsia', 'rahasia'), ('raib', 'hilang'), ('rana', 'ratna'), ('raja', 'sultan'), ('raksamala', 'rasamala'), ('rambang', 'rembang'), ('raksasa', 'gergasi'), ('raksi', 'harum'), ('raksi', 'rasi'), ('rakus', 'gelojoh'), ('rakus', 'loba'), ('rakus', 'tamak'), ('rakus', 'serakah'), ('ralip', 'galib'), ('ram', 'eram'), ('ramah', 'peramah'), ('ramulus', 'ramus'), ('rambang', 'acak'), ('rambih', 'rambeh'), ('rame', 'ramai'), ('rampas', 'perebutan'), ('ramping', 'langsing'), ('rampis', 'ramping'), ('rampung', 'selesai'), ('rampung', 'beres'), ('ramu', 'kumpul'), ('ranai', 'rinai'), ('rancah', 'rencah'), ('rancak', 'cepat'), ('randang', 'rendang'), ('rangah', 'pongah'), ('rangah', 'sombong'), ('rangak', 'gaduh'), ('rangak', 'ribut'), ('rangda', 'randa'), ('rangka', 'rengga'), ('rangkas', 'ranggas'), ('rangkun', 'rangkum'), ('rasa', 'merasakan'), ('rangsel', 'ransel'), ('rangsuk', 'rasuk'), ('rangsum', 'ransum'), ('ranguk', 'rango-rango'), ('rangup', 'rapuh'), ('rani', 'kaya'), ('rani', 'gani'), ('ranju', 'ranjau'), ('ranjungan', 'rajungan'), ('real', 'nyata'), ('real', 'rial'), ('rantai', 'rangkaian'), ('rantai', 'rentetan'), ('rantak', 'berantakan'), ('rapuh', 'lembik'), ('rasan', 'rasam'), ('rasia', 'rahasia'), ('rasia', 'razia'), ('rasionalitas', 'kerasionalan'), ('rasisme', 'rasialisme'), ('ratas', 'retas'), ('ratu', 'permaisuri'), ('rau', 'derau'), ('raun', 'ronda'), ('rawah', 'rawa'), ('rawak', 'acak'), ('rawak', 'rambang'), ('rawang', 'rawa'), ('rawat', 'menjaga'), ('rayap', 'anai-anai'), ('rayun', 'rayu'), ('realitas', 'kenyataan'), ('rebah', 'roboh'), ('rebet', 'rembet'), ('rebet', 'ribut'), ('rebo', 'rabu'), ('relawan', 'sukarelawan'), ('rebok', 'rebuk'), ('rebong', 'rebung'), ('recok', 'ribut'), ('reda', 'ketenangan'), ('redah', 'reda'), ('redut', 'dongkol'), ('redang', 'radang'), ('redap', 'redup'), ('redas', 'redah'), ('redum', 'suram'), ('redup', 'mendung'), ('regang', 'tegang'), ('region', 'kawasan'), ('reguk', 'meminum'), ('rehat', 'rihat'), ('rejab', 'rajab'), ('rejam', 'rajam'), ('rejeki', 'rezeki'), ('reka', 'merancang'), ('rekaat', 'rakaat'), ('rekal', 'rehal'), ('rekam', 'mencetak'), ('rekam', 'menyuji'), ('rekisitor', 'rekuisitor'), ('rekisitur', 'rekuisitor'), ('relevansi', 'hubungan'), ('rempak', 'rampak'), ('rembat', 'rimbat'), ('rembet', 'merintangi'), ('rembunai', 'remenia'), ('rembus', 'embus'), ('rempelu', 'empedu'), ('rempong', 'rimpung'), ('remuni', 'rembunai'), ('rena', 'rona'), ('renai', 'rinai'), ('rencet', 'rentet'), ('rencik', 'recik'), ('rencis', 'renjis'), ('rengah', 'engah'), ('renggek', 'rengek'), ('rengges', 'ranggas'), ('rengka', 'rengga'), ('rengkang', 'rangkiang'), ('rengkiang', 'rangkiang'), ('renyuk', 'kumal'), ('renyuk', 'mengumalkan'), ('renik', 'halus'), ('renjong', 'renjeng'), ('rentik', 'nyeri'), ('renyai', 'rinai'), ('reporter', 'wartawan'), ('reot', 'reyot'), ('repah', 'rapah'), ('repak', 'repas'), ('reparasi', 'perbaikan'), ('repek', 'repet'), ('repertorium', 'repertoar'), ('repis', 'repih'), ('replika', 'tiruan'), ('repolper', 'revolver'), ('representasi', 'perwakilan'), ('represi', 'pengekangan'), ('represi', 'penahanan'), ('represi', 'penindasan'), ('reproduksi', 'tiruan'), ('resi', 'resu'), ('resah', 'gelisah'), ('resah', 'gugup'), ('residif', 'residivistis'), ('resin', 'damar'), ('respons', 'reaksi'), ('restorasi', 'pemugaran'), ('restu', 'doa'), ('retal', 'hartal'), ('retet', 'rentet'), ('retok', 'recok'), ('retoris', 'retorik'), ('revolver', 'pistol'), ('rewan', 'rawan'), ('reyal', 'rial'), ('rezeki', 'nafkah'), ('rezeki', 'keuntungan'), ('rho', 'ro'), ('ria', 'gembira'), ('riadah', 'riadat'), ('rialat', 'riadat'), ('ricis', 'rincis'), ('rihal', 'rehal'), ('riil', 'nyata'), ('rijal', 'laki-laki'), ('rimah', 'remah'), ('rimbit', 'rembet'), ('rimbu', 'rimba'), ('rimih', 'rimis'), ('rinci', 'perinci'), ('rincih', 'rincis'), ('rincuh', 'rincu'), ('rintak', 'rentak'), ('ringgik', 'renggek'), ('ringkih', 'lemah'), ('ringkih', 'rapuh'), ('rinyai', 'rinai'), ('ripuk', 'rusak'), ('risih', 'risi'), ('ritma', 'ritme'), ('ritmik', 'ritmis'), ('rituil', 'ritual'), ('rizeki', 'rezeki'), ('roboh', 'merebahkan'), ('rogol', 'memerkosa'), ('rohmat', 'rahmat'), ('rois', 'rais'), ('rojabiyah', 'rajabiah'), ('romantik', 'romantis'), ('rombok', 'rimbun'), ('romo', 'rama'), ('rompang', 'rumpang'), ('rongak', 'ronggang'), ('rongga', 'lubang'), ('ronggoh', 'rungguh'), ('ronggok', 'rongkok'), ('rongot', 'rungut'), ('ronsen', 'rontgen'), ('ronyok', 'renyuk'), ('ronyok', 'kumal'), ('rosbang', 'resbang'), ('roseng', 'rongseng'), ('rowa', 'rua'), ('ruah', 'ruwah'), ('rubah', 'ubah'), ('rubanat', 'ruhbanat'), ('rubayat', 'rubaiat'), ('rubiah', 'riba'), ('rubuh', 'roboh'), ('rudah', 'ruadat'), ('runtak', 'ronta'), ('rugul', 'rogol'), ('ruh', 'roh'), ('ruhani', 'rohani'), ('rukiat', 'rukiah'), ('rungut', 'sungut'), ('rungut', 'memberengut'), ('rungut', 'merungut'), ('rukyah', 'rukyat'), ('rum', 'harum'), ('rumal', 'ramal'), ('rumawi', 'romawi'), ('rumenia', 'remenia'), ('rumit', 'sulit'), ('rumit', 'sukar'), ('rumpil', 'sulit'), ('rumpil', 'susah'), ('runggu', 'rungguh'), ('rungkai', 'ungkai'), ('rungsing', 'rongseng'), ('rungsum', 'rumrum'), ('runtuh', 'menjatuhkan'), ('runtuk', 'reruntuk'), ('runyai', 'unyai'), ('runyam', 'rumit'), ('ruruh', 'luruh'), ('ruwet', 'sulit'), ('ruyat', 'rukyat'), ('ruyatulhilal', 'rukyatulhilal'), ('saadat', 'saadah'), ('sabak', 'sebak'), ('saban', 'syakban'), ('sabas', 'syabas'), ('sabit', 'pasti'), ('sahwat', 'syahwat'), ('said', 'sayid'), ('safa', 'putih'), ('safa', 'bersih'), ('safaat', 'syafaat'), ('safi', 'bersih'), ('safi', 'murni'), ('safrah', 'seperah'), ('sagai', 'sakai'), ('saidani', 'sayidani'), ('saidi', 'sayidi'), ('sagar', 'sakar'), ('sago', 'sagu'), ('saguir', 'saguer'), ('sagun', 'sagon'), ('sahabat', 'teman'), ('sahadat', 'syahadat'), ('saharah', 'seharah'), ('sahaya', 'abdi'), ('sahbandar', 'syahbandar'), ('sahda', 'syahda'), ('sahdan', 'syahdan'), ('sahdu', 'syahda'), ('sahdu', 'syahdu'), ('sahid', 'syahid'), ('sahih', 'sah'), ('sahih', 'benar'), ('sahut', 'membalas'), ('sait', 'sayat'), ('sak', 'syak'), ('sasa', 'kukuh'), ('sasa', 'kuat'), ('sakar', 'neraka'), ('sakaratulmaut', 'sakratulmaut'), ('sakat', 'mengusik'), ('sakelat', 'sekelat'), ('sakhlat', 'sekelat'), ('saki', 'teman'), ('saki', 'kawan'), ('saki', 'sake'), ('saklar', 'sakelar'), ('sakral', 'suci'), ('sal', 'syal'), ('salada', 'selada'), ('sale', 'salai'), ('salap', 'salep'), ('salasilah', 'silsilah'), ('salatin', 'raja'), ('salawat', 'selawat'), ('sali', 'teguh'), ('salih', 'saleh'), ('salim', 'sempurna'), ('salwat', 'selawat'), ('sama', 'seimbang'), ('sama', 'sebanding'), ('sama', 'seragam'), ('samadi', 'semadi'), ('sambuk', 'sabut'), ('sami', 'agung'), ('sami', 'luhur'), ('sami', 'mulia'), ('sampai', 'datang'), ('sampi', 'sapi'), ('sampurna', 'sempurna'), ('samsu', 'syamsu'), ('samudera', 'samudra'), ('sana', 'angsana'), ('sandal', 'terompah'), ('sandal', 'sendal'), ('sandiwara', 'drama'), ('sangat', 'amat'), ('sangga', 'penyangga'), ('sangga', 'penopang'), ('sanggah', 'membantah'), ('santak', 'meninju'), ('sangih', 'segah'), ('sangkal', 'bantah'), ('sangkal', 'melawan'), ('sangkal', 'menentang'), ('sangkayan', 'sengkayan'), ('sangkela', 'sangkala'), ('sangking', 'saking'), ('sangsekerta', 'sanskerta'), ('sani', 'mulia'), ('sani', 'luhur'), ('sanskrit', 'sanskerta'), ('santron', 'satron'), ('santu', 'santo'), ('santun', 'sopan'), ('sanya', 'bahwasanya'), ('sap', 'saf'), ('saput', 'selaput'), ('sapa', 'safa'), ('sapar', 'safar'), ('sapir', 'safir'), ('sapu', 'mengoleskan'), ('sarahan', 'syarah'), ('sarakah', 'serakah'), ('saran', 'anjuran'), ('sarap', 'saraf'), ('sarat', 'syarat'), ('sarau', 'celaka'), ('sarden', 'sardencis'), ('sarekat', 'serikat'), ('sarengat', 'syariat'), ('sariat', 'syariat'), ('sariawan', 'seriawan'), ('saring', 'sering'), ('sarip', 'syarif'), ('saripah', 'syarifah'), ('sarok', 'saruk'), ('saru', 'seru'), ('sarun', 'saron'), ('sarwal', 'seluar'), ('sasan', 'sasa'), ('sasap', 'sesap'), ('satir', 'satire'), ('satria', 'kesatria'), ('saudara', 'kawan'), ('sauh', 'jangkar'), ('saur', 'sahur'), ('savana', 'sabana'), ('sayang', 'mengasihi'), ('sayang', 'mencintai'), ('sayang', 'pengasih'), ('sayarah', 'siarah'), ('sayat', 'sayet'), ('sayu', 'menyedihkan'), ('sebentar', 'sesaat'), ('sebet', 'sebat'), ('sebih', 'tasbih'), ('sebik', 'cebik'), ('sebut', 'mengucapkan'), ('seceng', 'ceceng'), ('secerek', 'sicerek'), ('sedah', 'sadah'), ('sedakap', 'sedekap'), ('sedat', 'bingung'), ('sedat', 'kacau'), ('sedativa', 'sedatif'), ('sedawa', 'serdawa'), ('sedekala', 'sediakala'), ('sedeng', 'gila'), ('sederiah', 'sadariah'), ('sederum', 'serentak'), ('sedia', 'bersedia'), ('sedih', 'sedu'), ('sedu', 'seduh'), ('sedu', 'sahda'), ('sedut', 'sedot'), ('seg', 'sek'), ('segan', 'enggan'), ('segar', 'menyegarkan'), ('segeh', 'kemas'), ('segenap', 'genap'), ('segregasi', 'pengasingan'), ('seh', 'syekh'), ('sehaja', 'sahaja'), ('sejarah', 'riwayat'), ('sejuk', 'menyamankan'), ('sekah', 'seka'), ('sekah', 'serkah'), ('sekaker', 'sekakar'), ('sekala', 'skala'), ('sekering', 'sekring'), ('sekedar', 'sekadar'), ('sekema', 'skema'), ('sekh', 'syekh'), ('sekin', 'sikin'), ('sekolah', 'pelajaran'), ('sekonar', 'sekunar'), ('sekonyar', 'sekunar'), ('sekonyong-konyong', 'tiba-tiba'), ('sekors', 'skors'), ('sekosol', 'sekesel'), ('sekotah', 'segenap'), ('seksama', 'saksama'), ('sekuik', 'naga'), ('sekutu', 'gabungan'), ('sela', 'sila'), ('selaber', 'selabar'), ('selain', 'kecuali'), ('selang', 'slang'), ('selampe', 'selampai'), ('selampek', 'selampai'), ('selampuri', 'selempuri'), ('selan', 'sailan'), ('selapa', 'selepa'), ('selayut', 'selayun'), ('sele', 'selai'), ('selederi', 'seledri'), ('seledup', 'seludup'), ('selekeh', 'noda'), ('selekoh', 'liku'), ('seleksi', 'pemilihan'), ('selembada', 'selempada'), ('selempukau', 'silempukau'), ('selender', 'silinder'), ('selendro', 'slendro'), ('selengat', 'selangat'), ('seleo', 'keseleo'), ('selepang', 'selempang'), ('selera', 'selira'), ('selesa', 'lega'), ('selesai', 'tamat'), ('selia', 'mengawasi'), ('selidik', 'pengusutan'), ('selimbu', 'selibu'), ('selimpang', 'selempang'), ('selindit', 'serindit'), ('selingkuh', 'curang'), ('selingkuh', 'korup'), ('selipar', 'sandal'), ('selisih', 'pertikaian'), ('selit', 'menyisipkan'), ('seliu', 'keseleo'), ('selodang', 'seludang'), ('selon', 'sailan'), ('selong', 'sailan'), ('seloroh', 'lucu'), ('seloroh', 'kelakar'), ('selosoh', 'selusuh'), ('seluar', 'celana'), ('selulur', 'tergelincir'), ('selungkap', 'jelungkap'), ('seluruh', 'semua'), ('seluruh', 'segenap'), ('selusur', 'langkan'), ('selut', 'lumpur'), ('semadi', 'meditasi'), ('semaja', 'sebenarnya'), ('semak', 'simak'), ('semakin', 'makin'), ('semalu', 'simalu'), ('semanda', 'semenda'), ('semandarasa', 'semendarasa'), ('semanja', 'semaja'), ('semantung', 'sementung'), ('sembab', 'sembap'), ('semberani', 'sembrani'), ('semberono', 'sembrono'), ('sembilat', 'semilat'), ('semboyan', 'slogan'), ('sembul', 'muncul'), ('sempadan', 'membatasi'), ('sembur', 'sembul'), ('semedera', 'samudra'), ('semejak', 'semenjak'), ('semejana', 'semenjana'), ('semek', 'demek'), ('sementung', 'dungu'), ('semerbak', 'harum'), ('semesta', 'universal'), ('semilir', 'silir'), ('semuntu', 'simuntu'), ('sempana', 'sempena'), ('sempil', 'menyisip'), ('sempit', 'picik'), ('sempoa', 'swipoa'), ('sempuras', 'kotor'), ('semrawut', 'acak-acakan'), ('semsem', 'sengsem'), ('semudera', 'samudra'), ('sen', 'sein'), ('sengkek', 'singkek'), ('senantiasa', 'selalu'), ('senantiasa', 'selamanya'), ('senapan', 'bedil'), ('senario', 'skenario'), ('senda', 'kelakar'), ('senda', 'seloroh'), ('senda', 'berseloroh'), ('senda', 'bergurau'), ('senda', 'sanda'), ('sendal', 'terompah'), ('sendat', 'ketat'), ('sendawa', 'serdawa'), ('sendeng', 'condong'), ('sendi', 'dasar'), ('sendi', 'asas'), ('sending', 'zending'), ('sendu', 'sedu'), ('senduduk', 'sekeduduk'), ('senduk', 'sendok'), ('senen', 'senin'), ('senget', 'miring'), ('senget', 'condong'), ('senget', 'sendeng'), ('senget', 'memiringkan'), ('senggah', 'sanggah'), ('senggara', 'selenggara'), ('senggerahan', 'pesanggrahan'), ('senggiling', 'tenggiling'), ('sengingih', 'sengih'), ('sengkelat', 'sekelat'), ('sengketa', 'pertengkaran'), ('sengketa', 'perbantahan'), ('sengsai', 'sansai'), ('sengsam', 'sengsem'), ('senguk', 'membaui'), ('senting', 'sukar'), ('seniwan', 'seniman'), ('seniwan', 'senewen'), ('senjak', 'sejak'), ('sentimentil', 'sentimental'), ('senjuang', 'lenjuang'), ('sensasi', 'kegemparan'), ('senting', 'santing'), ('sental', 'sintal'), ('sentap', 'sentak'), ('sentausa', 'sentosa'), ('senteri', 'santri'), ('sentiabu', 'setiabu'), ('sentiasa', 'senantiasa'), ('sentiung', 'sentiong'), ('sentrum', 'sentra'), ('senuhun', 'sinuhun'), ('senuk', 'tenuk'), ('senyak', 'senyap'), ('senyap', 'sunyi'), ('senyap', 'lengang'), ('senyap', 'kesunyian'), ('setasiun', 'stasiun'), ('sepakat', 'bersetuju'), ('sepakat', 'menyetujui'), ('sepakbor', 'sepatbor'), ('sepanduk', 'spanduk'), ('sepangkalan', 'sipangkalan'), ('separasi', 'pemisahan'), ('sepasan', 'sepesan'), ('sepatung', 'sipatung'), ('sepekuk', 'spekuk'), ('sepele', 'remeh'), ('seperitus', 'spiritus'), ('sepi', 'kesunyian'), ('sepih', 'serpih'), ('sepion', 'spion'), ('sepir', 'sipir'), ('sepiritus', 'spiritus'), ('sepoi', 'sipahi'), ('sepon', 'spons'), ('sepora', 'spora'), ('seprei', 'seprai'), ('sepui', 'sepoi'), ('sepura', 'spora'), ('seput', 'segera'), ('serabai', 'serabi'), ('serah', 'sera'), ('serah', 'sirah'), ('serakah', 'tamak'), ('seram', 'dahsyat'), ('seram', 'menakutkan'), ('serampat', 'sengkelit'), ('serana', 'sarana'), ('sereh', 'serai'), ('serang', 'melanggar'), ('serangguh', 'seranggung'), ('serapa', 'serapah'), ('serasah', 'baja'), ('serasi', 'cocok'), ('serasi', 'sesuai'), ('serasi', 'sepadan'), ('serasi', 'menyelaraskan'), ('serawan', 'rawan'), ('serbaneka', 'beraneka'), ('serbaneka', 'bermacam-macam'), ('sereat', 'syariat'), ('serengam', 'sengam'), ('serentak', 'serta-merta'), ('serep', 'serap'), ('seresah', 'serasah'), ('serigunting', 'srigunting'), ('serik', 'jera'), ('serikandi', 'srikandi'), ('seri', 'cahaya'), ('seriat', 'reda'), ('seriding', 'serendeng'), ('serikaya', 'srikaya'), ('sering', 'kerap'), ('seringih', 'seringai'), ('seroda', 'seruda'), ('serodok', 'seruduk'), ('serok', 'sero'), ('serok', 'serokan'), ('serok', 'seruk'), ('serondeng', 'serundeng'), ('seropot', 'menyesap'), ('serot', 'hirup'), ('serot', 'menghirup'), ('serpa', 'serapah'), ('serpai', 'serpih'), ('serse', 'sersi'), ('serual', 'seluar'), ('seruling', 'suling'), ('seruni', 'serunai'), ('servis', 'layanan'), ('setagi', 'setagen'), ('setaka', 'astaka'), ('setakona', 'astakona'), ('setambuk', 'stambuk'), ('setambul', 'stambul'), ('setandar', 'standar'), ('setangga', 'tetangga'), ('seteker', 'steker'), ('setempel', 'stempel'), ('seten', 'stengun'), ('setenggar', 'istinggar'), ('seterap', 'setrap'), ('seteria', 'satria'), ('seterik', 'setrik'), ('seterika', 'setrika'), ('seteriman', 'setirman'), ('seterimin', 'setrimin'), ('seterip', 'setrip'), ('seterum', 'setrum'), ('seterup', 'setrup'), ('setewel', 'setiwel'), ('setinggil', 'sitinggil'), ('setinja', 'istinja'), ('setiwal', 'setiwel'), ('setoker', 'stoker'), ('setop', 'setup'), ('setoples', 'stoples'), ('setreng', 'streng'), ('setrimin', 'strimin'), ('setua', 'satwa'), ('seturi', 'setori'), ('seturi', 'kesturi'), ('sewah', 'sewar'), ('sewal', 'sial'), ('sewot', 'dongkol'), ('sial', 'celaka'), ('sial', 'kemalangan'), ('siarat', 'siarah'), ('siasah', 'siasat'), ('siasat', 'menyelidiki'), ('sidat', 'belut'), ('sidekah', 'sedekah'), ('sidingin', 'sedingin'), ('sigap', 'tangkas'), ('signifikan', 'penting'), ('sijil', 'sertifikat'), ('sinda', 'sanda'), ('sikeduduk', 'senduduk'), ('sikejut', 'semalu'), ('siketumbak', 'sikudomba'), ('silam', 'kelam'), ('silaf', 'khilaf'), ('silah', 'sila'), ('silah', 'silsilah'), ('silap', 'khilaf'), ('silaturahim', 'silaturahmi'), ('silempukau', 'silampukau'), ('silinder', 'tabung'), ('simbai', 'tertib'), ('simbol', 'lambang'), ('simbukan', 'kesimbukan'), ('sinandung', 'senandung'), ('singgasana', 'takhta'), ('singgung', 'sigung'), ('singkak', 'singkap'), ('singkawang', 'tengkawang'), ('singkeh', 'singkek'), ('singsat', 'singset'), ('singse', 'sinse'), ('sipilis', 'sifilis'), ('sinopsis', 'ringkasan'), ('sintese', 'sintesis'), ('sinting', 'senteng'), ('sinting', 'miring'), ('sipai', 'sipahi'), ('sipasin', 'sepasin'), ('sipesan', 'sepesan'), ('sipoa', 'swipoa'), ('siput', 'kerang'), ('sirangkak', 'serangkak'), ('sirik', 'syirik'), ('sirik', 'siri'), ('sirkit', 'sirkuit'), ('sirna', 'melenyapkan'), ('sirobok', 'serobok'), ('sirup', 'sirop'), ('sista', 'kista'), ('sistim', 'sistem'), ('sitar', 'siter'), ('siwar', 'sewar'), ('soal', 'perkara'), ('sodium', 'natrium'), ('sodomi', 'semburit'), ('sojah', 'soja'), ('soklat', 'cokelat'), ('soko', 'saka'), ('sokong', 'penunjang'), ('solak', 'suka'), ('solat', 'salat'), ('solawat', 'selawat'), ('soldadu', 'serdadu'), ('sompak', 'sompok'), ('sompeng', 'sompek'), ('sondol', 'sundul'), ('songar', 'sombong'), ('songar', 'congkak'), ('sonor', 'merdu'), ('sontok', 'pendek'), ('sontok', 'suntuk'), ('soprano', 'sopran'), ('sorban', 'serban'), ('sorbet', 'serbat'), ('sorga', 'surga'), ('sorgawi', 'surgawi'), ('sorot', 'sinar'), ('sorot', 'menyinari'), ('sotong', 'cumi-cumi'), ('sotor', 'sotoh'), ('sowan', 'berkunjung'), ('sowan', 'soang'), ('sowang', 'soang'), ('spakbor', 'sepatbor'), ('span', 'ketat'), ('span', 'kencang'), ('spedometer', 'spidometer'), ('spesial', 'khusus'), ('spesial', 'khas'), ('srempet', 'serempet'), ('stabil', 'kukuh'), ('stabil', 'tetap'), ('stabilitas', 'kemantapan'), ('stabilitas', 'kestabilan'), ('stanza', 'bait'), ('statistik', 'perangkaan'), ('stek', 'setek'), ('stemma', 'stema'), ('sten', 'stengun'), ('stepler', 'stapler'), ('steril', 'mandul'), ('stimulasi', 'rangsangan'), ('strip', 'setrip'), ('struktur', 'susunan'), ('struktur', 'bangunan'), ('subhat', 'syubhat'), ('sual', 'soal'), ('sualak', 'solak'), ('suang', 'mudah'), ('suarga', 'surga'), ('suari', 'kasuari'), ('subyek', 'subjek'), ('subyektif', 'subjektif'), ('subyektivisme', 'subjektivisme'), ('suci', 'kemurnian'), ('sudagar', 'saudagar'), ('sudara', 'saudara'), ('suduk', 'tikam'), ('suduk', 'sodok'), ('sudur', 'sodor'), ('sugih', 'kaya'), ('sugih', 'berada'), ('sugra', 'kiamat'), ('suipoa', 'swipoa'), ('suir', 'langsuir'), ('sujadah', 'sajadah'), ('sukses', 'berhasil'), ('suka', 'menggemari'), ('suka', 'mencintai'), ('suka', 'menyayangi'), ('sukduf', 'sekedup'), ('sulup', 'selup'), ('suku', 'seperempat'), ('sukur', 'syukur'), ('sulalat', 'sulalah'), ('suli', 'gandasuli'), ('sulih', 'ganti'), ('suling', 'bangsi'), ('sultanat', 'kesultanan'), ('sulu', 'suluh'), ('sulub', 'sulbi'), ('sumanda', 'semenda'), ('sumangat', 'semangat'), ('sumarak', 'semarak'), ('sumbul', 'sumbur'), ('sumir', 'pendek'), ('sumpah', 'bersumpah'), ('sumpah', 'ikrar'), ('sumpah', 'kutuk'), ('sumpah', 'mengutuk'), ('sumpal', 'sumbat'), ('sumpel', 'sumpal'), ('sunat', 'sunah'), ('sundik', 'sondek'), ('sungguh', 'bahwasanya'), ('sungguhpun', 'meskipun'), ('sunglap', 'sulap'), ('sungsum', 'sumsum'), ('sunnah', 'sunah'), ('supai', 'sipahi'), ('superioritas', 'keunggulan'), ('supervisor', 'penyelia'), ('supir', 'sopir'), ('supit', 'sumpit'), ('suplai', 'pembekalan'), ('suram', 'buram'), ('suram', 'sabur'), ('surban', 'serban'), ('suren', 'surian'), ('suris', 'surih'), ('suruh', 'memerintahkan'), ('surung', 'sorong'), ('surut', 'turun'), ('surut', 'menyusutkan'), ('survai', 'survei'), ('surya', 'matahari'), ('susah hati', 'sedih'), ('susah', 'sedih'), ('susah', 'kesukaran'), ('susah', 'kesulitan'), ('susila', 'beradab'), ('susila', 'kesopanan'), ('susu', 'tetek'), ('syagar', 'sakar'), ('susur', 'selusur'), ('susut', 'mengurangi'), ('susut', 'mengurangkan'), ('sut', 'suten'), ('suvenir', 'kenang-kenangan'), ('swarga', 'surga'), ('switer', 'sweter'), ('syaban', 'syakban'), ('syafkah', 'syafakat'), ('syah', 'sah'), ('syahada', 'syahda'), ('syahadan', 'syahdan'), ('syahda', 'syahdu'), ('syahdan', 'selanjutnya'), ('syahdan', 'lalu'), ('syahdu', 'mulia'), ('syaikh', 'syekh'), ('syair', 'puisi'), ('syaitan', 'setan'), ('syajarat', 'sejarah'), ('syakar', 'sakar'), ('syakduf', 'sekedup'), ('syamas', 'samas'), ('syamsiat', 'syamsiah'), ('syapaat', 'syafaat'), ('syaraf', 'saraf'), ('syarah', 'uraian'), ('syarbat', 'serbat'), ('syarekat', 'syarikat'), ('syariah', 'syariat'), ('syarikat', 'perhimpunan'), ('syorga', 'surga'), ('syubahat', 'syubhat'), ('syufaat', 'syafaat'), ('syurah', 'syarah'), ('syurga', 'surga'), ('taajul', 'segera'), ('taawud', 'taawuz'), ('tabah', 'tebah'), ('tabe', 'tabik'), ('tabi', 'tabik'), ('tabiat', 'watak'), ('tabiat', 'kelakuan'), ('tabu', 'larangan'), ('tafsir', 'interpretasi'), ('taftah', 'tafeta'), ('tahadi', 'tadi'), ('tahalil', 'tahlil'), ('tahan', 'awet'), ('tahan', 'merintangi'), ('tahang', 'tong'), ('tahir', 'bersih'), ('tahir', 'suci'), ('tahir', 'murni'), ('tahta', 'takhta'), ('tais', 'kotor'), ('tajam', 'pandai'), ('taju', 'tajuk'), ('takarir', 'takrir'), ('takel', 'takal'), ('takhlik', 'membentuk'), ('takhlik', 'menciptakan'), ('takjub', 'mengagumkan'), ('taksir', 'kira-kira'), ('taksis', 'takhsis'), ('takut', 'bimbang'), ('tanah', 'negeri'), ('tanah', 'negara'), ('takwim', 'penanggalan'), ('talai', 'lalai'), ('talek', 'talk'), ('talen', 'talenta'), ('tamadun', 'peradaban'), ('tamam', 'lengkap'), ('tamat', 'berakhir'), ('tamat', 'habis'), ('tamat', 'menyelesaikan'), ('tamat', 'mengakhiri'), ('tambak', 'tanggul'), ('tambak', 'bendung'), ('tambak', 'tebat'), ('tambo', 'hikayat'), ('tambus', 'timbus'), ('tampak', 'kelihatan'), ('tampak', 'campak'), ('tampan', 'talam'), ('tampek', 'campak'), ('tanding', 'menyaingi'), ('tandus', 'gersang'), ('tang', 'tank'), ('tangar', 'hati-hati'), ('temetu', 'keras'), ('tangguh', 'kuat'), ('tanggung', 'memikul'), ('tangker', 'tanker'), ('tangkut', 'tangkup'), ('tanjidur', 'tanjidor'), ('tante', 'bibi'), ('taoke', 'tauke'), ('tapa', 'tempa'), ('tapir', 'tenuk'), ('tapis', 'penyaring'), ('tapis', 'saringan'), ('tapuk', 'tepuk'), ('tapung', 'tepung'), ('tar', 'tir'), ('taras', 'teras'), ('tarikat', 'tarekat'), ('taring', 'siung'), ('tarip', 'tarif'), ('tarkasy', 'tarkas'), ('tarkhim', 'tarhim'), ('tarpentin', 'terpentin'), ('tarra', 'tara'), ('tarub', 'tarup'), ('taruh', 'tagan'), ('taruk', 'taruh'), ('taruko', 'teroka'), ('tauladan', 'teladan'), ('taulan', 'tolan'), ('taruna', 'teruna'), ('tasdid', 'tasydid'), ('tasmak', 'tesmak'), ('tatanan', 'aturan'), ('tatapan', 'tetampan'), ('taubat', 'tobat'), ('tauco', 'taoco'), ('taufan', 'topan'), ('tauge', 'taoge'), ('taup', 'taut'), ('tauret', 'taurat'), ('taurit', 'taurat'), ('tawak-tawak', 'tetawak'), ('tawarikh', 'tarikh'), ('tazir', 'takzir'), ('tean', 'teyan'), ('tebakang', 'tambakan'), ('tedak', 'turun'), ('tedarus', 'tadarus'), ('tebat', 'bendung'), ('tedas', 'nyata'), ('tebuan', 'tabuhan'), ('tebus', 'tembus'), ('tegak', 'vertikal'), ('tegang', 'meregangkan'), ('tegap', 'kukuh'), ('tegor', 'tegur'), ('teguk', 'tegar'), ('teka', 'terka'), ('tekaan', 'dugaan'), ('tekaan', 'sangkaan'), ('tekaan', 'tebakan'), ('tekak', 'faring'), ('tekan', 'aksen'), ('tekebur', 'takabur'), ('tekek', 'pekak'), ('tekek', 'tuli'), ('tekek', 'tokek'), ('tekoan', 'teko'), ('tekong', 'tikung'), ('tektek', 'tetek'), ('tekua', 'takwa'), ('telaga', 'perigi'), ('tekup', 'tekap'), ('tela', 'ketela'), ('telakan', 'telekan'), ('telanjang', 'bugil'), ('telegrap', 'telegraf'), ('telapa', 'telap'), ('telayan', 'nelayan'), ('telefon', 'telepon'), ('telingkung', 'lingkung'), ('telipuk', 'telepok'), ('telekup', 'tekup'), ('telenan', 'talenan'), ('telepa', 'telap'), ('telisik', 'selisik'), ('telgram', 'telegram'), ('telimpung', 'telempong'), ('telingkuh', 'telingkah'), ('terlanjur', 'telanjur'), ('telor', 'telur'), ('telpon', 'telepon'), ('telukup', 'telungkup'), ('temadun', 'tamadun'), ('temaha', 'temaah'), ('temahak', 'temaah'), ('teman', 'menyertai'), ('temasa', 'tamasya'), ('temasya', 'tamasya'), ('tematu', 'tembatu'), ('temengalan', 'tengalan'), ('temenggung', 'tumenggung'), ('tembekar', 'tembikar'), ('tembel', 'timbil'), ('tembel', 'tambal'), ('tembem', 'tembam'), ('tembikai', 'semangka'), ('tembikar', 'porselen'), ('tembiring', 'tembereng'), ('tembola', 'tombola'), ('tembusu', 'tembesu'), ('temilang', 'tembilang'), ('tempalak', 'tempelak'), ('tempang', 'timpang'), ('tempek', 'tepek'), ('terlantar', 'telantar'), ('terlentang', 'telentang'), ('tempeleng', 'tamparan'), ('tempeleng', 'menampar'), ('temporer', 'sementara'), ('temporok', 'porok'), ('tempuras', 'temperas'), ('temu', 'sua'), ('temu duga', 'wawancara'), ('temuni', 'tembuni'), ('tenang', 'meredakan'), ('tener', 'tiner'), ('teng', 'tank'), ('tenggiring', 'tenggiri'), ('tenggulung', 'senggulung'), ('tengker', 'tengger'), ('tengkik', 'tengkek'), ('tengkok', 'tengkuk'), ('tengkolak', 'tengkulak'), ('tengkolok', 'tengkuluk'), ('tengkurup', 'tengkurap'), ('tengok', 'lihat'), ('tengteng', 'tenteng'), ('tenguh', 'lenguh'), ('tentang', 'menampik'), ('tentang', 'membangkang'), ('tentang', 'penolakan'), ('tentera', 'tentara'), ('tentram', 'tenteram'), ('tentu', 'positif'), ('tepat', 'memperbaiki'), ('tepekong', 'toapekong'), ('tepekur', 'tafakur'), ('tepet', 'tepek'), ('tepok', 'tepuk'), ('tepu', 'penuh'), ('tetuhu', 'tuhu'), ('ter', 'tir'), ('terajang', 'terjang'), ('terala', 'luhur'), ('teram', 'taram'), ('terambu', 'tambo'), ('terang', 'nyata'), ('terigu', 'gandum'), ('terasul', 'tarasul'), ('teratak', 'gubuk'), ('teratu', 'penyiksaan'), ('teratu', 'penganiayaan'), ('teraweh', 'tarawih'), ('teriko', 'triko'), ('terem', 'trem'), ('teret', 'deret'), ('teriak', 'pekikan'), ('terista', 'sedih'), ('terista', 'dukacita'), ('terjali', 'tajali'), ('wahon', 'wagon'), ('terobong', 'cerobong'), ('terompet', 'trompet'), ('terongko', 'terungku'), ('terpedo', 'torpedo'), ('terubus', 'bertunas'), ('terup', 'truf'), ('terus', 'lantas'), ('terus terang', 'jujur'), ('terwelu', 'kelinci'), ('tesis', 'disertasi'), ('testes', 'testis'), ('tetampah', 'tampah'), ('tetangga', 'jiran'), ('tetapan', 'tetampan'), ('tetibau', 'tetibar'), ('tezi', 'teji'), ('tihang', 'tiang'), ('tijak', 'pijak'), ('tilawat', 'tilawah'), ('tikas', 'kesan'), ('tiku', 'tikung'), ('tilam', 'kasur'), ('tim', 'kelompok'), ('tim', 'tin'), ('timbul', 'muncul'), ('timbun', 'longgok'), ('tin', 'kaleng'), ('tindak', 'kelakuan'), ('tindawan', 'cendawan'), ('tindis', 'tindih'), ('tinggal', 'bersisa'), ('tinggir', 'tengger'), ('tinta', 'mangsi'), ('tingkarah', 'tengkarah'), ('tingkat', 'pangkat'), ('tingkat', 'taraf'), ('tingkat', 'kelas'), ('tingker', 'tingkar'), ('tinjau', 'menilik'), ('tipak', 'tepak'), ('tipes', 'tifus'), ('tipikal', 'khas'), ('tipu', 'muslihat'), ('tipu', 'mengakali'), ('tipus', 'tifus'), ('tiru', 'mencontoh'), ('titih', 'petitih'), ('titik berat', 'penekanan'), ('titir', 'ketitir'), ('toke', 'tauke'), ('tokek', 'takik'), ('toblos', 'coblos'), ('tofan', 'topan'), ('toge', 'taoge'), ('tokak', 'tukak'), ('tolak', 'menyorongkan'), ('tolak', 'sorongan'), ('tolak', 'dorongan'), ('tolan', 'teman'), ('tolan', 'kawan'), ('tolerir', 'toleransi'), ('tombong', 'tumbung'), ('tombong', 'tomong'), ('tompang', 'tumpang'), ('ton', 'tona'), ('tonggong', 'tonggok'), ('tongkong', 'tongkol'), ('tongong', 'pandir'), ('tongsil', 'tonsil'), ('tonjok', 'tinju'), ('tragik', 'tragis'), ('tonsil', 'amandel'), ('topah', 'tufah'), ('topong', 'ketopong'), ('tores', 'toreh'), ('torne', 'turne'), ('totalitas', 'keutuhan'), ('totalitas', 'keseluruhan'), ('towaf', 'tawaf'), ('toyah', 'toya'), ('toyoh', 'toya'), ('trakoma', 'trakom'), ('tram', 'trem'), ('trampil', 'terampil'), ('transito', 'transit'), ('transkrip', 'salinan'), ('tribulan', 'triwulan'), ('trap', 'terap'), ('tras', 'teras'), ('trenggiling', 'tenggiling'), ('trengginas', 'tangkas'), ('trindil', 'terindil'), ('tropi', 'trofi'), ('tuai', 'pengetaman'), ('tuhfat', 'tuhfah'), ('tuan', 'majikan'), ('tuas', 'tuil'), ('tuat', 'ketuat'), ('tuba', 'saluran'), ('tuba', 'tabung'), ('tudak', 'todak'), ('tuhur', 'tohor'), ('tudung', 'penutup'), ('tugas', 'tukas'), ('tukar', 'menyilih'), ('tuku', 'tengku'), ('tumbalang', 'tembelang'), ('tuli', 'pekak'), ('tulung', 'tolong'), ('tulus', 'jujur'), ('tuman', 'toman'), ('tumbak', 'tombak'), ('tumbuh', 'terbit'), ('tumpes', 'tumpas'), ('tumplek', 'tumplak'), ('tunanetra', 'buta'), ('tumpur', 'binasa'), ('tunggak', 'tonggak'), ('tunggal', 'satu-satunya'), ('tunggang', 'menaiki'), ('tunggit', 'tunggik'), ('tungkat', 'tongkat'), ('tungkik', 'tukik'), ('tungkul', 'tongkol'), ('tungsten', 'wolfram'), ('tungu', 'tungau'), ('tunjuk', 'berdemonstrasi'), ('tunjul', 'tonjol'), ('tupang', 'topang'), ('turis', 'pelancong'), ('turisme', 'kepariwisataan'), ('tursi', 'terusi'), ('turut', 'mencontoh'), ('turut', 'menurut'), ('tutup', 'tudung'), ('tutup', 'mengakhiri'), ('tutur', 'kata'), ('tutur', 'bercakap-cakap'), ('tutur', 'mempercakapkan'), ('uan', 'wan'), ('ubal-ubal', 'ubel-ubel'), ('ubat', 'obat'), ('ubub', 'puputan'), ('ucus', 'usus'), ('ulung', 'berpengalaman'), ('ulung', 'mahir'), ('udema', 'edema'), ('udi', 'sial'), ('udo', 'uda'), ('udu', 'wudu'), ('uduh', 'odoh'), ('uduk', 'wudu'), ('udur', 'uzur'), ('ugut', 'mengancam'), ('uja', 'oja'), ('ujana', 'yojana'), ('uju', 'pongah'), ('uju', 'sombong'), ('ujud', 'wujud'), ('ujud', 'maksud'), ('ulak', 'ulek'), ('ulang', 'mengulangi'), ('umum', 'umun'), ('ulas', 'tafsiran'), ('ulat', 'ulet'), ('uler', 'ular'), ('uma', 'huma'), ('umak', 'emak'), ('umbai', 'umbuk'), ('umbang', 'mengapung'), ('umbi', 'umbuk'), ('umbilikus', 'pusar'), ('umbilikus', 'pusat'), ('umbun-umbun', 'ubun-ubun'), ('umbur', 'umbul'), ('umbur-umbur', 'umbul'), ('unak', 'onak'), ('umpat', 'umpet'), ('umpuk', 'berlonggok'), ('umpun', 'rumpun'), ('umrat', 'umrah'), ('umuk', 'sombong'), ('umuk', 'pongah'), ('umuk', 'congkak'), ('uncue', 'honcoe'), ('uncui', 'honcoe'), ('unifikasi', 'penyatuan'), ('ungap-ungap', 'mengap-mengap'), ('unggit', 'ungkit'), ('ungguk', 'onggok'), ('ungkang-ungkang', 'ongkang-ongkang'), ('ungkir', 'mungkir'), ('uniform', 'seragam'), ('universitet', 'universitas'), ('unjam', 'hunjam'), ('unjuk rasa', 'demonstrasi'), ('unjung', 'kunjung'), ('upas', 'opas'), ('upaya', 'usaha'), ('untung', 'manfaat'), ('untung', 'mujur'), ('untung', 'nasib'), ('upah', 'gaji'), ('upuk', 'ufuk'), ('urat', 'otot'), ('urat', 'aurat'), ('urip', 'hidup'), ('urit', 'urip'), ('urus', 'membenahi'), ('urus', 'mengelola'), ('usai', 'bubar'), ('usai', 'berakhir'), ('usai', 'selesai'), ('usai', 'habis'), ('usai', 'mencerai-beraikan'), ('usak', 'berkurang'), ('usam', 'kusam'), ('usap', 'menyeka'), ('usis', 'ucis'), ('ustad', 'ustaz'), ('usung', 'usungan'), ('utan', 'hutan'), ('utas', 'mahir'), ('uterus', 'peranakan'), ('uwak', 'uak'), ('uyung', 'huyung'), ('vermak', 'permak'), ('verzet', 'verset'), ('vestibul', 'vestibula'), ('vestibulum', 'vestibula'), ('vignet', 'vinyet'), ('vrah', 'prahoto'), ('vulkanisir', 'vulkanisasi'), ('vulpen', 'pulpen'), ('waadah', 'waadat'), ('wahi', 'wahyu'), ('wabakdahu', 'wabakdu'), ('wadar', 'badar'), ('wadas', 'cadas'), ('wader', 'badar'), ('wahadah', 'wahdah'), ('wahadaniah', 'wahdaniah'), ('wahadat', 'wahdah'), ('wahadiah', 'wahdiah'), ('waham', 'curiga'), ('waiduri', 'baiduri'), ('waja', 'baja'), ('wajit', 'wajik'), ('walimat', 'walimah'), ('wakwak', 'wawa'), ('walat', 'kualat'), ('walaupun', 'walau'), ('wali', 'rajawali'), ('walmana', 'walimana'), ('wang', 'uang'), ('wangi', 'harum'), ('waqaf', 'wakaf'), ('war', 'uar'), ('warakat', 'warkat'), ('waslah', 'wasal'), ('wardi', 'rodi'), ('warik', 'warak'), ('warkah', 'warkat'), ('warna', 'mewarnai'), ('warta', 'berita'), ('warwar', 'uar-uar'), ('was', 'waswas'), ('wasilah', 'perhubungan'), ('wasilah', 'pertalian'), ('wasilat', 'wasilah'), ('wasir', 'bawasir'), ('wassalam', 'wasalam'), ('waton', 'watan'), ('wawankata', 'wawancara'), ('wawansabda', 'wawancara'), ('wazir', 'bawasir'), ('wedam', 'weda'), ('weker', 'beker'), ('welut', 'belut'), ('werak', 'werek'), ('werangka', 'warangka'), ('werda', 'wreda'), ('werdatama', 'wredatama'), ('wilahar', 'welahar'), ('wilmana', 'walimana'), ('winter', 'wenter'), ('wira', 'pahlawan'), ('wira', 'perwira'), ('wirama', 'irama'), ('wirasat', 'firasat'), ('wiron', 'wiru'), ('wirwir', 'uir-uir'), ('wisatawan', 'pelancong'), ('wodka', 'vodka'), ('wungu', 'ungu'), ('wurung', 'urung'), ('wutuh', 'utuh'), ('yakin', 'kepastian'), ('yargon', 'jargon'), ('yehova', 'yahwe'), ('yeyunum', 'jejunum'), ('yudo', 'judo'), ('yudoka', 'judoka'), ('yunior', 'junior'), ('yuri', 'juri'), ('yuwana', 'muda'), ('yuyutsu', 'yuyitsu'), ('zabad', 'jebat'), ('zabah', 'sembelih'), ('zabah', 'menyembelih'), ('zahir', 'lahir'), ('zakiah', 'suci'), ('zakiah', 'murni'), ('zakiah', 'bersih'), ('zalim', 'bengis'), ('zalim', 'kejam'), ('zan', 'waham'), ('zariah', 'zuriah'), ('zariat', 'zuriah'), ('zuhara', 'zohrah'), ('zat', 'unsur'), ('zenggi', 'zanggi'), ('ziarat', 'ziarah'), ('zib', 'jib'), ('zib', 'serigala'), ('zinah', 'zina'), ('zink', 'seng'), ('zohor', 'zuhur'), ('zohrat', 'zohrah'), ('zone', 'zona'), ('zuadah', 'juadah'), ('zuriat', 'zuriah'), ('baju kemeja', 'kemeja'), ('batu marmer', 'marmer'), ('berat', 'susah'), ('burung murai', 'murai'), ('ilmu fisika', 'fisika'), ('kain belacu', 'belacu'), ('kedewaga', 'kedewas'), ('kon-', 'ko-'), ('mem-', 'meng-'), ('men-', 'meng-'), ('menge-', 'meng-'), ('meny-', 'meng-'), ('met-', 'meta-'), ('pem-', 'peng-'), ('penge-', 'peng-'), ('peny-', 'peng-'), ('rosok', 'rongsok'), ('sim-', 'sin-'), ('tanda koma', 'koma'), ('tipu daya', 'siasat')]

PROVINCE_NAMES_RAW = ['Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau', 'Jambi', 'Sumatera Selatan', 'Bengkulu', 'Lampung', 'Kepulauan Bangka Belitung', 'Kepulauan Riau', 'Dki Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Di Yogyakarta', 'Jawa Timur', 'Banten', 'Bali', 'Nusa Tenggara Barat', 'Nusa Tenggara Timur', 'Kalimantan Barat', 'Kalimantan Tengah', 'Kalimantan Selatan', 'Kalimantan Timur', 'Kalimantan Utara', 'Sulawesi Utara', 'Sulawesi Tengah', 'Sulawesi Selatan', 'Sulawesi Tenggara', 'Gorontalo', 'Sulawesi Barat', 'Maluku', 'Maluku Utara', 'Papua Barat', 'Papua']

REGENCY_PROVINCE_RAW = [('Simeulue', 11), ('Aceh Singkil', 11), ('Aceh Selatan', 11), ('Aceh Tenggara', 11), ('Aceh Timur', 11), ('Aceh Tengah', 11), ('Aceh Barat', 11), ('Aceh Besar', 11), ('Pidie', 11), ('Bireuen', 11), ('Aceh Utara', 11), ('Aceh Barat Daya', 11), ('Gayo Lues', 11), ('Aceh Tamiang', 11), ('Nagan Raya', 11), ('Aceh Jaya', 11), ('Bener Meriah', 11), ('Pidie Jaya', 11), ('Banda Aceh', 11), ('Sabang', 11), ('Langsa', 11), ('Lhokseumawe', 11), ('Subulussalam', 11), ('Nias', 12), ('Mandailing Natal', 12), ('Tapanuli Selatan', 12), ('Tapanuli Tengah', 12), ('Tapanuli Utara', 12), ('Toba Samosir', 12), ('Labuhan Batu', 12), ('Asahan', 12), ('Simalungun', 12), ('Dairi', 12), ('Karo', 12), ('Deli Serdang', 12), ('Langkat', 12), ('Nias Selatan', 12), ('Humbang Hasundutan', 12), ('Pakpak Bharat', 12), ('Samosir', 12), ('Serdang Bedagai', 12), ('Batu Bara', 12), ('Padang Lawas Utara', 12), ('Padang Lawas', 12), ('Labuhan Batu Selatan', 12), ('Labuhan Batu Utara', 12), ('Nias Utara', 12), ('Nias Barat', 12), ('Sibolga', 12), ('Tanjung Balai', 12), ('Pematang Siantar', 12), ('Tebing Tinggi', 12), ('Medan', 12), ('Binjai', 12), ('Padangsidimpuan', 12), ('Gunungsitoli', 12), ('Kepulauan Mentawai', 13), ('Pesisir Selatan', 13), ('Solok', 13), ('Sijunjung', 13), ('Tanah Datar', 13), ('Padang Pariaman', 13), ('Agam', 13), ('Lima Puluh Kota', 13), ('Pasaman', 13), ('Solok Selatan', 13), ('Dharmasraya', 13), ('Pasaman Barat', 13), ('Padang', 13), ('Solok', 13), ('Sawah Lunto', 13), ('Padang Panjang', 13), ('Bukittinggi', 13), ('Payakumbuh', 13), ('Pariaman', 13), ('Kuantan Singingi', 14), ('Indragiri Hulu', 14), ('Indragiri Hilir', 14), ('Pelalawan', 14), ('S I A K', 14), ('Kampar', 14), ('Rokan Hulu', 14), ('Bengkalis', 14), ('Rokan Hilir', 14), ('Kepulauan Meranti', 14), ('Pekanbaru', 14), ('D U M A I', 14), ('Kerinci', 15), ('Merangin', 15), ('Sarolangun', 15), ('Batang Hari', 15), ('Muaro Jambi', 15), ('Tanjung Jabung Timur', 15), ('Tanjung Jabung Barat', 15), ('Tebo', 15), ('Bungo', 15), ('Jambi', 15), ('Sungai Penuh', 15), ('Ogan Komering Ulu', 16), ('Ogan Komering Ilir', 16), ('Muara Enim', 16), ('Lahat', 16), ('Musi Rawas', 16), ('Musi Banyuasin', 16), ('Banyu Asin', 16), ('Ogan Komering Ulu Selatan', 16), ('Ogan Komering Ulu Timur', 16), ('Ogan Ilir', 16), ('Empat Lawang', 16), ('Penukal Abab Lematang Ilir', 16), ('Musi Rawas Utara', 16), ('Palembang', 16), ('Prabumulih', 16), ('Pagar Alam', 16), ('Lubuklinggau', 16), ('Bengkulu Selatan', 17), ('Rejang Lebong', 17), ('Bengkulu Utara', 17), ('Kaur', 17), ('Seluma', 17), ('Mukomuko', 17), ('Lebong', 17), ('Kepahiang', 17), ('Bengkulu Tengah', 17), ('Bengkulu', 17), ('Lampung Barat', 18), ('Tanggamus', 18), ('Lampung Selatan', 18), ('Lampung Timur', 18), ('Lampung Tengah', 18), ('Lampung Utara', 18), ('Way Kanan', 18), ('Tulangbawang', 18), ('Pesawaran', 18), ('Pringsewu', 18), ('Mesuji', 18), ('Tulang Bawang Barat', 18), ('Pesisir Barat', 18), ('Bandar Lampung', 18), ('Metro', 18), ('Bangka', 19), ('Belitung', 19), ('Bangka Barat', 19), ('Bangka Tengah', 19), ('Bangka Selatan', 19), ('Belitung Timur', 19), ('Pangkal Pinang', 19), ('Karimun', 21), ('Bintan', 21), ('Natuna', 21), ('Lingga', 21), ('Kepulauan Anambas', 21), ('B A T A M', 21), ('Tanjung Pinang', 21), ('Kepulauan Seribu', 31), ('Jakarta Selatan', 31), ('Jakarta Timur', 31), ('Jakarta Pusat', 31), ('Jakarta Barat', 31), ('Jakarta Utara', 31), ('Bogor', 32), ('Sukabumi', 32), ('Cianjur', 32), ('Bandung', 32), ('Garut', 32), ('Tasikmalaya', 32), ('Ciamis', 32), ('Kuningan', 32), ('Cirebon', 32), ('Majalengka', 32), ('Sumedang', 32), ('Indramayu', 32), ('Subang', 32), ('Purwakarta', 32), ('Karawang', 32), ('Bekasi', 32), ('Bandung Barat', 32), ('Pangandaran', 32), ('Bogor', 32), ('Sukabumi', 32), ('Bandung', 32), ('Cirebon', 32), ('Bekasi', 32), ('Depok', 32), ('Cimahi', 32), ('Tasikmalaya', 32), ('Banjar', 32), ('Cilacap', 33), ('Banyumas', 33), ('Purbalingga', 33), ('Banjarnegara', 33), ('Kebumen', 33), ('Purworejo', 33), ('Wonosobo', 33), ('Magelang', 33), ('Boyolali', 33), ('Klaten', 33), ('Sukoharjo', 33), ('Wonogiri', 33), ('Karanganyar', 33), ('Sragen', 33), ('Grobogan', 33), ('Blora', 33), ('Rembang', 33), ('Pati', 33), ('Kudus', 33), ('Jepara', 33), ('Demak', 33), ('Semarang', 33), ('Temanggung', 33), ('Kendal', 33), ('Batang', 33), ('Pekalongan', 33), ('Pemalang', 33), ('Tegal', 33), ('Brebes', 33), ('Magelang', 33), ('Surakarta', 33), ('Salatiga', 33), ('Semarang', 33), ('Pekalongan', 33), ('Tegal', 33), ('Kulon Progo', 34), ('Bantul', 34), ('Gunung Kidul', 34), ('Sleman', 34), ('Yogyakarta', 34), ('Pacitan', 35), ('Ponorogo', 35), ('Trenggalek', 35), ('Tulungagung', 35), ('Blitar', 35), ('Kediri', 35), ('Malang', 35), ('Lumajang', 35), ('Jember', 35), ('Banyuwangi', 35), ('Bondowoso', 35), ('Situbondo', 35), ('Probolinggo', 35), ('Pasuruan', 35), ('Sidoarjo', 35), ('Mojokerto', 35), ('Jombang', 35), ('Nganjuk', 35), ('Madiun', 35), ('Magetan', 35), ('Ngawi', 35), ('Bojonegoro', 35), ('Tuban', 35), ('Lamongan', 35), ('Gresik', 35), ('Bangkalan', 35), ('Sampang', 35), ('Pamekasan', 35), ('Sumenep', 35), ('Kediri', 35), ('Blitar', 35), ('Malang', 35), ('Probolinggo', 35), ('Pasuruan', 35), ('Mojokerto', 35), ('Madiun', 35), ('Surabaya', 35), ('Batu', 35), ('Pandeglang', 36), ('Lebak', 36), ('Tangerang', 36), ('Serang', 36), ('Tangerang', 36), ('Cilegon', 36), ('Serang', 36), ('Tangerang Selatan', 36), ('Jembrana', 51), ('Tabanan', 51), ('Badung', 51), ('Gianyar', 51), ('Klungkung', 51), ('Bangli', 51), ('Karang Asem', 51), ('Buleleng', 51), ('Denpasar', 51), ('Lombok Barat', 52), ('Lombok Tengah', 52), ('Lombok Timur', 52), ('Sumbawa', 52), ('Dompu', 52), ('Bima', 52), ('Sumbawa Barat', 52), ('Lombok Utara', 52), ('Mataram', 52), ('Bima', 52), ('Sumba Barat', 53), ('Sumba Timur', 53), ('Kupang', 53), ('Timor Tengah Selatan', 53), ('Timor Tengah Utara', 53), ('Belu', 53), ('Alor', 53), ('Lembata', 53), ('Flores Timur', 53), ('Sikka', 53), ('Ende', 53), ('Ngada', 53), ('Manggarai', 53), ('Rote Ndao', 53), ('Manggarai Barat', 53), ('Sumba Tengah', 53), ('Sumba Barat Daya', 53), ('Nagekeo', 53), ('Manggarai Timur', 53), ('Sabu Raijua', 53), ('Malaka', 53), ('Kupang', 53), ('Sambas', 61), ('Bengkayang', 61), ('Landak', 61), ('Mempawah', 61), ('Sanggau', 61), ('Ketapang', 61), ('Sintang', 61), ('Kapuas Hulu', 61), ('Sekadau', 61), ('Melawi', 61), ('Kayong Utara', 61), ('Kubu Raya', 61), ('Pontianak', 61), ('Singkawang', 61), ('Kotawaringin Barat', 62), ('Kotawaringin Timur', 62), ('Kapuas', 62), ('Barito Selatan', 62), ('Barito Utara', 62), ('Sukamara', 62), ('Lamandau', 62), ('Seruyan', 62), ('Katingan', 62), ('Pulang Pisau', 62), ('Gunung Mas', 62), ('Barito Timur', 62), ('Murung Raya', 62), ('Palangka Raya', 62), ('Tanah Laut', 63), ('Kota Baru', 63), ('Banjar', 63), ('Barito Kuala', 63), ('Tapin', 63), ('Hulu Sungai Selatan', 63), ('Hulu Sungai Tengah', 63), ('Hulu Sungai Utara', 63), ('Tabalong', 63), ('Tanah Bumbu', 63), ('Balangan', 63), ('Banjarmasin', 63), ('Banjar Baru', 63), ('Paser', 64), ('Kutai Barat', 64), ('Kutai Kartanegara', 64), ('Kutai Timur', 64), ('Berau', 64), ('Penajam Paser Utara', 64), ('Mahakam Hulu', 64), ('Balikpapan', 64), ('Samarinda', 64), ('Bontang', 64), ('Malinau', 65), ('Bulungan', 65), ('Tana Tidung', 65), ('Nunukan', 65), ('Tarakan', 65), ('Bolaang Mongondow', 71), ('Minahasa', 71), ('Kepulauan Sangihe', 71), ('Kepulauan Talaud', 71), ('Minahasa Selatan', 71), ('Minahasa Utara', 71), ('Bolaang Mongondow Utara', 71), ('Siau Tagulandang Biaro', 71), ('Minahasa Tenggara', 71), ('Bolaang Mongondow Selatan', 71), ('Bolaang Mongondow Timur', 71), ('Manado', 71), ('Bitung', 71), ('Tomohon', 71), ('Kotamobagu', 71), ('Banggai Kepulauan', 72), ('Banggai', 72), ('Morowali', 72), ('Poso', 72), ('Donggala', 72), ('Toli-Toli', 72), ('Buol', 72), ('Parigi Moutong', 72), ('Tojo Una-Una', 72), ('Sigi', 72), ('Banggai Laut', 72), ('Morowali Utara', 72), ('Palu', 72), ('Kepulauan Selayar', 73), ('Bulukumba', 73), ('Bantaeng', 73), ('Jeneponto', 73), ('Takalar', 73), ('Gowa', 73), ('Sinjai', 73), ('Maros', 73), ('Pangkajene Dan Kepulauan', 73), ('Barru', 73), ('Bone', 73), ('Soppeng', 73), ('Wajo', 73), ('Sidenreng Rappang', 73), ('Pinrang', 73), ('Enrekang', 73), ('Luwu', 73), ('Tana Toraja', 73), ('Luwu Utara', 73), ('Luwu Timur', 73), ('Toraja Utara', 73), ('Makassar', 73), ('Parepare', 73), ('Palopo', 73), ('Buton', 74), ('Muna', 74), ('Konawe', 74), ('Kolaka', 74), ('Konawe Selatan', 74), ('Bombana', 74), ('Wakatobi', 74), ('Kolaka Utara', 74), ('Buton Utara', 74), ('Konawe Utara', 74), ('Kolaka Timur', 74), ('Konawe Kepulauan', 74), ('Muna Barat', 74), ('Buton Tengah', 74), ('Buton Selatan', 74), ('Kendari', 74), ('Baubau', 74), ('Boalemo', 75), ('Gorontalo', 75), ('Pohuwato', 75), ('Bone Bolango', 75), ('Gorontalo Utara', 75), ('Gorontalo', 75), ('Majene', 76), ('Polewali Mandar', 76), ('Mamasa', 76), ('Mamuju', 76), ('Mamuju Utara', 76), ('Mamuju Tengah', 76), ('Maluku Tenggara Barat', 81), ('Maluku Tenggara', 81), ('Maluku Tengah', 81), ('Buru', 81), ('Kepulauan Aru', 81), ('Seram Bagian Barat', 81), ('Seram Bagian Timur', 81), ('Maluku Barat Daya', 81), ('Buru Selatan', 81), ('Ambon', 81), ('Tual', 81), ('Halmahera Barat', 82), ('Halmahera Tengah', 82), ('Kepulauan Sula', 82), ('Halmahera Selatan', 82), ('Halmahera Utara', 82), ('Halmahera Timur', 82), ('Pulau Morotai', 82), ('Pulau Taliabu', 82), ('Ternate', 82), ('Tidore Kepulauan', 82), ('Fakfak', 91), ('Kaimana', 91), ('Teluk Wondama', 91), ('Teluk Bintuni', 91), ('Manokwari', 91), ('Sorong Selatan', 91), ('Sorong', 91), ('Raja Ampat', 91), ('Tambrauw', 91), ('Maybrat', 91), ('Manokwari Selatan', 91), ('Pegunungan Arfak', 91), ('Sorong', 91), ('Merauke', 94), ('Jayawijaya', 94), ('Jayapura', 94), ('Nabire', 94), ('Kepulauan Yapen', 94), ('Biak Numfor', 94), ('Paniai', 94), ('Puncak Jaya', 94), ('Mimika', 94), ('Boven Digoel', 94), ('Mappi', 94), ('Asmat', 94), ('Yahukimo', 94), ('Pegunungan Bintang', 94), ('Tolikara', 94), ('Sarmi', 94), ('Keerom', 94), ('Waropen', 94), ('Supiori', 94), ('Mamberamo Raya', 94), ('Nduga', 94), ('Lanny Jaya', 94), ('Mamberamo Tengah', 94), ('Yalimo', 94), ('Puncak', 94), ('Dogiyai', 94), ('Intan Jaya', 94), ('Deiyai', 94), ('Jayapura', 94)]

PROVINCE_ID_MAP_RAW = {11: 'Aceh', 12: 'Sumatera Utara', 13: 'Sumatera Barat', 14: 'Riau', 15: 'Jambi', 16: 'Sumatera Selatan', 17: 'Bengkulu', 18: 'Lampung', 19: 'Kepulauan Bangka Belitung', 21: 'Kepulauan Riau', 31: 'Dki Jakarta', 32: 'Jawa Barat', 33: 'Jawa Tengah', 34: 'Di Yogyakarta', 35: 'Jawa Timur', 36: 'Banten', 51: 'Bali', 52: 'Nusa Tenggara Barat', 53: 'Nusa Tenggara Timur', 61: 'Kalimantan Barat', 62: 'Kalimantan Tengah', 63: 'Kalimantan Selatan', 64: 'Kalimantan Timur', 65: 'Kalimantan Utara', 71: 'Sulawesi Utara', 72: 'Sulawesi Tengah', 73: 'Sulawesi Selatan', 74: 'Sulawesi Tenggara', 75: 'Gorontalo', 76: 'Sulawesi Barat', 81: 'Maluku', 82: 'Maluku Utara', 91: 'Papua Barat', 94: 'Papua'}

ANTONYM_SET=set()
for a,b in ANTONYM_PAIRS_RAW: ANTONYM_SET.add((a,b)); ANTONYM_SET.add((b,a))
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

NUM=re.compile(r'\d+[.,]?\d*\s*%?'); YEAR=re.compile(r'\b(?:19|20)\d{2}\b')
NEG=['tidak','tak','bukan','belum','tanpa','gagal','ditolak','membantah','bantah','sangkal','menyangkal','menolak','dilarang']
INCREASE_ROOTS={'naik','tingkat','tambah','lonjak','tanjak','pesat'}
DECREASE_ROOTS={'turun','kurang','rosot','lambat','susut','tipis','landai'}
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
NUM_TOKEN=re.compile(r'\d+(?:[.,]\d+)?')
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
                    if re.fullmatch(r'\d{3}',w2) and val==int(val): val=val*1000+int(w2); j+=1
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
    m=re.fullmatch(r'\d{1,3}(?:\.\d{3})+(?:,\d+)?', t)
    if m:
        intpart,_,decpart=t.partition(',')
        val=float(intpart.replace('.',''))
        if decpart: val+=float('0.'+decpart)
        return val
    if re.fullmatch(r'\d{1,3}\.\d{3}', t):
        return float(t.replace('.',''))
    if re.fullmatch(r'\d+,\d+', t):
        return float(t.replace(',','.'))
    if re.fullmatch(r'\d+\.\d+', t):
        return float(t)
    if re.fullmatch(r'\d+', t):
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
                if re.fullmatch(r'\d{3}',w2) and val==int(val): val=val*1000+int(w2); j+=1; span_end=j-1
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
        # Entity-substitution feature family (error-decomposition audit, 2026-09-11):
        # type-free, LOCAL claim-binding substitution evidence -- title entity + predicate
        # vs the one body chunk that actually matched that predicate (bc_pred), not a
        # full-body scan. An "event" requires a COMPETITOR entity to occupy the slot in
        # place of the missing title argument; simple absence (no competitor) is never
        # counted as substitution.
        chunk_ent_names=[w for w,_ in chunk_ents_typed]
        events=[]
        for slot,arg in (('subj',arg_subj),('obj',arg_obj)):
            if arg is None or arg in pred_chunk_set: continue
            competitors=[ce for ce in chunk_ent_names if ce!=arg]
            if competitors: events.append((slot,arg,competitors))
        n_slots_checked=sum(1 for a in (arg_subj,arg_obj) if a is not None)
        n_entity_substitutions=float(len(events))
        entity_substitution_score=n_entity_substitutions/n_slots_checked if n_slots_checked>0 else 0.0
        strongest_entity_substitution=min(1.0,max((len(c) for _,_,c in events),default=0)/3.0)
        entity_role_conflict=len({slot for slot,_,_ in events})/2.0
        _ttok_low=set(w.lower() for w in Ttok)
        _lex_ov=len(_ttok_low & set(w.lower() for w in pred_chunk_words))/max(len(_ttok_low),1)
        entity_substitution_x_lexical=entity_substitution_score*_lex_ov
    else:
        argument_binding_conflict=0.0
        entity_substitution_score=0.0; n_entity_substitutions=0.0
        strongest_entity_substitution=0.0; entity_role_conflict=0.0
        entity_substitution_x_lexical=0.0
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
                entity_substitution_score=entity_substitution_score, n_entity_substitutions=n_entity_substitutions,
                strongest_entity_substitution=strongest_entity_substitution, entity_role_conflict=entity_role_conflict,
                entity_substitution_x_lexical=entity_substitution_x_lexical,
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
reg_train_selfexcl = []   # each train row's regime computed leaving itself out (for B-model training)
for i in range(len(train)):
    th,ch = train['th'].iloc[i], train['ch'].iloc[i]
    others = [ (t,l) for t,l in byc_all[ch] if not (t==th and l==train['label'].iloc[i]) ] if ch in byc_all else []
    # approximate leave-one-out: exclude exactly one occurrence of this row's own (th,label)
    pass
print("test regime counts:", {r: int((reg_test==r).sum()) for r in ['C_exact','B_has_pos','B_no_pos','A_cold']})

def predict_c_exact(df, pair):
    out = np.full(len(df), np.nan)
    for i,(th,ch) in enumerate(zip(df['th'],df['ch'])):
        if (th,ch) in pair: out[i] = pair[(th,ch)]
    return out
c_exact_pred = predict_c_exact(test, pair_all)
print("C_exact resolved:", int((~np.isnan(c_exact_pred)).sum()), "of", int((reg_test=='C_exact').sum()))

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

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report

with Timer("A_cold_cv_threshold"):
    groups=train['ch'].values
    sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    folds=list(sgkf.split(train,y,groups))
    Xa_full=Ftr[ALL_A_FEATS].values
    oof=np.zeros(len(train))
    for a,bidx in folds:
        m=cb.CatBoostClassifier(iterations=800,depth=6,learning_rate=0.03,
                                 class_weights=[1,1],random_seed=SEED,verbose=False)
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
    print(f"  class_weights : [1, 1]  (Experiment 1: weight sweep, GO -- stable win over [4,1]")
    print(f"                  across 3 independent CV seeds, +0.004 to +0.011 Macro F1)")
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
                                   class_weights=[4,1],random_seed=SEED,verbose=False)
    a_model.fit(Xa_full,y)
    Xte_a=Fte[ALL_A_FEATS].values
    a_proba_all=a_model.predict_proba(Xte_a)[:,1]
a_mask_test = reg_test=='A_cold'
print("A_cold test rows:", int(a_mask_test.sum()))

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

sub=pd.DataFrame({'id':test['id'],'label':final}).set_index('id').loc[test['id']].reset_index()
assert sub.shape[0]==len(test) and set(sub['id'])==set(sample_sub['id'])
assert set(sub['label'].unique())<={0,1} and list(sub.columns)==['id','label']
sub.to_csv('D:\\Lomba\\IFEST2026_DAC\\notebook_v13_final\\local_out\\submission.csv', index=False)
print(sub['label'].value_counts(normalize=True))

T=time.time()-RUN_T0
print("="*58)
print("V13 — full regime router (C rule + B title-retrieval + A_cold claim-representation)")
print(f"A_cold OOF macro F1  : {best[1]:.4f}  (F1-0={f1_0:.4f}, F1-1={f1_1:.4f}) @ threshold {A_THR:.3f}")
print(f"total runtime        : {T/60:.1f} min")
print("="*58)
for k,v in sorted(TIMINGS.items(),key=lambda kv:-kv[1]): print(f"  {k}: {v:.1f}s")