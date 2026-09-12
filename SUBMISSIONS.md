# Submissions

Pulled directly from `kaggle competitions submissions -c penyisihan-ifest-2026-dac`.
Not all of these were made by an agent session — flagged per row.

| date | file | description | public score | origin |
|---|---|---|---|---|
| 2026-09-05 17:56 | submission.csv | — | 0.73629 | v2/v3, **non-compliant** (IndoBERT) |
| 2026-09-05 19:42 | submission.csv | — | 0.75591 | v4, **non-compliant** |
| 2026-09-05 21:14 | submission.csv | v5: hybrid retrieval + contrastive + hard-neg mining | 0.74366 | **non-compliant** |
| 2026-09-06 05:43 | submission.csv | v6: synthetic entity-swap augmentation | 0.73477 | **non-compliant** |
| 2026-09-06 07:14 | submission_v5_plus_rules.csv | v5 model + same-content rule layer | **0.88169** | **non-compliant** (base model is v5/IndoBERT) — best score on record, but disqualifiable |
| 2026-09-06 12:12 | submission.csv | — | 0.86061 | unclear provenance, predates the v7+ compliant rebuild timestamps — never fully identified |
| 2026-09-08 06:54 | submission.csv | — | 0.47345 | **unknown — not from any agent session script** |
| 2026-09-08 20:26 | submission.csv | v12: v11 + structural multi-granularity chunk-matching | 0.85374 | agent session |
| 2026-09-08 20:53 | submission.csv | v12b: + robust multi-split weighted threshold (fixes v12's single-split bias) | 0.85374 | agent session |
| 2026-09-08 21:05 | submission.csv | — | ERROR | agent session (failed upload) |
| 2026-09-08 21:09 | submission.csv | — | 0.78435 | **unknown — not from any agent session script** |
| 2026-09-08 22:38 | submission.csv | — | 0.81901 | **unknown — not from any agent session script** |
| 2026-09-09 02:44 | submission.csv | — | 0.94383 | **unknown — not from any agent session script** |
| 2026-09-09 02:46 | submission(1).csv | — | 0.93532 | **unknown — not from any agent session script** |
| 2026-09-09 05:58 | submission.csv | — | **0.96428** | **unknown — not from any agent session script** |
| 2026-09-11 | submission.csv | v13: full regime router (C rule + B title-retrieval + A_cold 71-feat CatBoost) — **with train/serve bug**: final fit `class_weights=[4,1]` while threshold was calibrated on `[1,1]` OOF | ~0.808 | agent session, compliant. Score as reported by user; pull exact value from Kaggle |
| 2026-09-11 | submission.csv | v13, same pipeline, **bug fixed** (`A_CLASS_WEIGHTS` shared by CV loop and final fit); 185 rows flipped 0→1, predicted positive rate 0.863→0.915 | **~0.870** | agent session, compliant. **Best compliant score on record.** Score as reported by user; pull exact value from Kaggle |

## Unresolved

Six submissions between 2026-09-08 06:54 and 2026-09-09 05:58 (scores 0.47–0.96) were not
produced by any script this project's agent sessions ran. Their existence is recorded here
factually rather than omitted or guessed at. If these came from manual work outside the
agent sessions, worth reconciling with `EXPERIMENTS.md` — a real 0.96 would contradict
every compliant-pipeline ceiling estimated in this repo and should be investigated, not
assumed to be either legitimate or an error.

## What's actually validated and compliant

The full compliant pipeline (`notebook_v13_final/ifest2026_dac_v13_final.ipynb`, generated
by `build_notebook.py`) has been run end-to-end and submitted: **~0.870 public LB**
(2026-09-11). Per-regime honest CV behind it: `C_exact` rule 1.0, `B` title-retrieval
≈0.985, `A_cold` 0.6995 (seed 42; 5-seed mean 0.6909). The earlier analytical estimate of
≈0.857 is superseded.

The gap between the first v13 submission (~0.808) and the second (~0.870) is one bug —
the final A_cold fit used different `class_weights` than the CV that chose its threshold —
not a modelling change. CV could not see it. Worth remembering: **when LB and honest CV
disagree, audit train/serve consistency before touching features.**
