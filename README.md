# TRACE — Detecting Forged News Headlines Without Pretrained Models

DAC IFest 2026, Qualifying Round — Team cupuu. Detects whether a COVID-19 news headline
matches its own article, when forged headlines borrow their replacement words from that
same article and the competition rules ban every pretrained language model.

## Open this first
- [`Penyisihan_DAC2026_cupuu.ipynb`](./Penyisihan_DAC2026_cupuu.ipynb) — **the final,
  submitted notebook.** Start here; it's self-contained and documents the full TRACE
  (Token-level Replacement And Consistency Ensemble) pipeline end to end.

Every other file/folder in this repo (`notebook_v2` … `notebook_v13_final`, the
`diag_*.py` / `*_log.txt` files) is exploratory or diagnostic work from earlier iterations
— useful as a paper trail, not something a reader needs to open.

## Result
**OOF Macro-F1 0.9578** (95% bootstrap CI 0.9522–0.9637), vs a 0.513 TF-IDF + logistic
regression baseline. Built from scratch: TF-IDF, BM25, LSA, and word2vec are all trained
on this competition's own text, with zero pretrained models, per the competition rules.

No dataset is committed here.
