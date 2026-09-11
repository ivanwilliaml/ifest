"""Build an .xlsx summarizing model bottlenecks with real, manually-verified example rows
(title, content, label, problem), ~5 rows per problem category. Sourced entirely from the
user's own manual labels in labeling_set_labeled.csv (Indonesian news headline/body pairs,
IFEST 2026 DAC competition data)."""
import pandas as pd

df = pd.read_csv(r'D:\Downloads\labeling_set_labeled.csv')
hi = df[df['priority'].str.startswith('HIGH')].copy()

def row(id_, problem):
    r = hi[hi['id'] == id_].iloc[0]
    return dict(title=r['title'], content=r['body'], label=int(r['true_label']), problem=problem)

out_rows = []

# 1. Label noise / dataset construction ceiling
support = hi[hi['relation_type'] == 'support'].head(5)
for _, r in support.iterrows():
    out_rows.append(dict(
        category='1. Label noise (dataset)',
        title=r['title'], content=r['body'], label=int(r['true_label']),
        problem=f"Body tampak MENDUKUNG judul (baca manual), tapi label dataset = 0 (tidak konsisten). "
                f"Kemungkinan noise dari proses tampering, atau butuh headline asli artikel (tidak tersedia untuk baris cold) untuk memastikan. Catatan: {r['notes']}"
    ))

# 2. Entity substitution -- Person/Pejabat
person_ids = ['tr00009', 'tr01283', 'tr02575', 'tr03545', 'tr07599', 'tr09111', 'tr09518', 'tr11605', 'tr13180', 'tr13686', 'tr14316']
for id_ in person_ids[:5]:
    r = hi[hi['id'] == id_].iloc[0]
    out_rows.append(dict(
        category='2. Entity substitution - Person/Pejabat',
        title=r['title'], content=r['body'], label=int(r['true_label']),
        problem=f"Judul mengatribusikan pernyataan/aksi ke orang yang BEDA dari yang disebut body. "
                f"Model tidak punya cara mengenali 'siapa subjek dari predikat' tanpa dependency parser -- "
                f"nama disebut di body (jadi 'entity ada'), tapi bukan sebagai pelaku aksi yang sama. {r['notes']}"
    ))

# 3. Entity substitution -- Geografi/Lokasi
geo_ids = ['tr00415', 'tr01798', 'tr01916', 'tr02122', 'tr05533', 'tr06131', 'tr06763', 'tr06817', 'tr07112', 'tr10516', 'tr10726', 'tr10775', 'tr11645', 'tr12140', 'tr12962']
for id_ in geo_ids[:5]:
    r = hi[hi['id'] == id_].iloc[0]
    out_rows.append(dict(
        category='3. Entity substitution - Geografi/Lokasi',
        title=r['title'], content=r['body'], label=int(r['true_label']),
        problem=f"Kota/provinsi/fasilitas (bandara, RS, dsb.) di judul BEDA dari yang disebut body. "
                f"Kadang di luar cakupan gazetteer administratif (bandara/RS/ponpes bukan kabupaten/provinsi resmi), "
                f"kadang gazetteer-nya ada tapi model tidak cukup mengandalkan sinyalnya. {r['notes']}"
    ))

# 4. Entity substitution -- Organisasi/Brand/Negara
org_ids = ['tr00117', 'tr01903', 'tr01952', 'tr03412', 'tr05828', 'tr06448', 'tr08015', 'tr11958', 'tr12366', 'tr14213']
for id_ in org_ids[:5]:
    r = hi[hi['id'] == id_].iloc[0]
    out_rows.append(dict(
        category='4. Entity substitution - Organisasi/Brand/Negara',
        title=r['title'], content=r['body'], label=int(r['true_label']),
        problem=f"Organisasi/institusi/merek/negara di judul BEDA dari body (mis. Kemenhan vs Kemenhub -- "
                f"singkatan mirip, gampang salah baca bahkan oleh model). Sering di luar gazetteer ORG yang "
                f"sengaja dijaga kecil (perluasan gazetteer ORG terbukti -0.0068 di eksperimen sebelumnya). {r['notes']}"
    ))

# 5. Quantity substitution
qty = hi[(hi['relation_type'] == 'contradict') & (hi['contradiction_subtype'] == 'quantity')].head(5)
for _, r in qty.iterrows():
    out_rows.append(dict(
        category='5. Quantity substitution',
        title=r['title'], content=r['body'], label=int(r['true_label']),
        problem=f"Angka di judul tidak cocok dengan angka relevan di body (entitas/topik sama, angka beda). {r['notes']}"
    ))

# 6. Polarity / negation contradiction
pol = hi[(hi['relation_type'] == 'contradict') & (hi['contradiction_subtype'] == 'polarity')]
for _, r in pol.iterrows():
    out_rows.append(dict(
        category='6. Polarity/Negation contradiction',
        title=r['title'], content=r['body'], label=int(r['true_label']),
        problem=f"Arah/polaritas klaim berlawanan (mis. 'akan' vs 'tidak akan', 'naik' vs 'turun') tapi tidak "
                f"tertangkap fitur negasi/polaritas yang ada -- biasanya karena predicate-nya beda kata "
                f"(bukan window kata yang sama). {r['notes']}"
    ))

# 7. Event/predicate mismatch
ev = hi[(hi['relation_type'] == 'contradict') & (hi['contradiction_subtype'] == 'event_predicate')]
for _, r in ev.iterrows():
    out_rows.append(dict(
        category='7. Event/Predicate mismatch',
        title=r['title'], content=r['body'], label=int(r['true_label']),
        problem=f"Entitas sama, tapi KEJADIAN/AKSI yang diklaim judul berbeda dari yang terjadi di body "
                f"(bukan soal entity atau angka, murni soal apa yang sebenarnya terjadi). {r['notes']}"
    ))

out_df = pd.DataFrame(out_rows)[['category', 'title', 'content', 'label', 'problem']]
out_path = r'D:\Lomba\IFEST2026_DAC\notebook_v13_final\model_bottlenecks.xlsx'
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    out_df.to_excel(writer, index=False, sheet_name='bottlenecks')
    ws = writer.sheets['bottlenecks']
    widths = {'A': 32, 'B': 45, 'C': 70, 'D': 8, 'E': 70}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for cell in ws[1]:
        cell.font = cell.font.copy(bold=True)

print(f"Saved {len(out_df)} rows across {out_df['category'].nunique()} categories to {out_path}")
print(out_df['category'].value_counts())
