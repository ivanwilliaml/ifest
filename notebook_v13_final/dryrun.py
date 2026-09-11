"""Execute the v13 notebook locally by rewriting Kaggle paths (fixed path-rewrite, no drive-root bug)."""
import io, os, json

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v13_final.ipynb'), encoding='utf-8'))
src = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
src = src.replace("/kaggle/input", r"D:\Lomba\IFEST2026_DAC\data".replace("\\", "\\\\"))
src = src.replace("/kaggle/working/submission.csv", os.path.join(d, 'local_out', 'submission.csv').replace("\\", "\\\\"))
os.makedirs(os.path.join(d, 'local_out'), exist_ok=True)
out = os.path.join(d, '_dryrun.py')
io.open(out, 'w', encoding='utf-8').write(src)
exec(compile(src, out, 'exec'), {'__name__': '__main__'})
