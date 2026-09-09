import io, os, json, nbformat as nbf

d = os.path.dirname(os.path.abspath(__file__))
src = io.open(os.path.join(d, '..', 'neural_scratch_exp.py'), encoding='utf-8').read()

# argparse -> fixed config for the kernel
src = src.split("ap = argparse.ArgumentParser()")[0] + """
class _A: pass
A = _A()
A.pos, A.epochs, A.dim, A.vocab = 4000, 12, 128, 20000
A.tlen, A.clen, A.bs = 32, 256, 64
""" + src.split("if A.quick:\n    A.pos, A.epochs, A.dim, A.clen, A.vocab = 800, 2, 64, 128, 5000\n")[1]

cells = [
    nbf.v4.new_markdown_cell(
        "# From-scratch neural experiment — no pretrained weights anywhere\n\n"
        "Rules allow *\"embedding yang dilatih dari awal menggunakan data kompetisi\"*. "
        "Every parameter here is randomly initialised and trained only on this "
        "competition's training rows. No tokenizer, vocabulary, embedding or weight "
        "is downloaded.\n\n"
        "**Question:** does a from-scratch neural encoder beat — or add to — the "
        "classical TF-IDF control on **cold-start rows**, the only regime a model decides?\n\n"
        "**Subsampling is asymmetric:** all 1,437 negatives are kept, positives are "
        "downsampled. Negatives are the scarce class; cutting them would make a null "
        "result uninterpretable.\n\n"
        "**Decision rule, fixed before running:** neural passes only if it beats the "
        "classical control on the same split by >= +0.01 macro F1."),
    nbf.v4.new_code_cell(src),
]
nb = nbf.v4.new_notebook(cells=cells)
nb['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb['metadata']['language_info'] = {'name': 'python', 'version': '3.11'}
p = os.path.join(d, 'ifest2026_dac_neural_scratch.ipynb')
nbf.write(nb, p)
print('written', p)

meta = {
    "id": "ivanwllm/ifest-2026-dac-neural-scratch",
    "title": "IFEST 2026 DAC neural scratch",
    "code_file": "ifest2026_dac_neural_scratch.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_internet": False,
    "dataset_sources": [], "competition_sources": ["penyisihan-ifest-2026-dac"],
    "kernel_sources": [],
}
io.open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8').write(
    json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
