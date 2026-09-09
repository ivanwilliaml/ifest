"""Execute the v11 notebook locally by rewriting the Kaggle input path."""
import io, os, json, sys

d = os.path.dirname(os.path.abspath(__file__))
nb = json.load(io.open(os.path.join(d, 'ifest2026_dac_v11.ipynb'), encoding='utf-8'))
src = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
src = src.replace("'/kaggle/input'", r"r'D:\Lomba\IFEST2026_DAC\data'")
src = src.replace("'/kaggle/working'", repr(os.path.join(d, 'local_out')))
os.makedirs(os.path.join(d, 'local_out'), exist_ok=True)
os.chdir(os.path.join(d, 'local_out'))
out = os.path.join(d, '_dryrun.py')
io.open(out, 'w', encoding='utf-8').write(src)
exec(compile(src, out, 'exec'), {'__name__': '__main__'})
