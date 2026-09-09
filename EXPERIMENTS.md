# Experiments

Written by hand from the session history (no automated lineage tooling in this repo —
see `CLAUDE.md` for why). Grouped by regime, since `C_exact` / `B_has_pos` / `B_no_pos` /
`A_cold` turned out to need genuinely different methods, not one shared model.

**Primary metric:** `macro_f1` (maximize). All scores below are honest cross-validation
(grouped by `content_hash` where regime membership requires it, repeated-split where the
regime needs the mixture to appear at all) — **never** the leaderboard score used for
selection. See `SUBMISSIONS.md` for the actual Kaggle history.

## Pre-session history (v2–v11, see `_diag_cache` and old `notebook_v*` dirs)

```mermaid
graph LR
  v2["v2/v3 hybrid retrieval<br/>+ IndoBERT cross-encoder<br/>LB 0.73629<br/>NON-COMPLIANT"]
  v4["v4 same-content contrastive<br/>LB 0.75591 (best v2-v6)<br/>NON-COMPLIANT"]
  v5["v5 + cross-article<br/>hard-negative mining<br/>LB 0.74366<br/>NON-COMPLIANT"]
  v5r["v5 + same-content rule layer<br/>LB 0.88169<br/>NON-COMPLIANT (rules discovered too late)"]
  v6["v6 synthetic entity-swap<br/>augmentation, LB 0.73477<br/>NON-COMPLIANT"]
  v8["v8_legal: first from-scratch<br/>rebuild, dense OOF 0.6457"]
  v9["v9 + tamper-target vocab<br/>(near-dup retrieval tested, FALSIFIED)<br/>dense OOF 0.6581"]
  v10["v10 + salience features<br/>dense OOF 0.6631"]
  v11["v11 + full salience suite<br/>rule layer finalized<br/>dense OOF 0.6631, blend 0.6684"]
  v2-->v4-->v5-->v5r
  v5-->v6
  v8-->v9-->v10-->v11
```

Everything from `v8` on is compliant (no pretrained weights). Rule-C-copy and
`B_no_pos`-rule were ablated OFF `v11` onward (worth −0.0016 and 0.0000 respectively —
see `C_exact` section below for why that number is misleading in isolation).

## `C_exact` — exact (title, body) pair already in train (25.7% of test)

| method | Macro F1 | note |
|---|---|---|
| rule: copy known label | **1.0** | provably correct — 0 label noise ever observed in exact pairs |
| deferred to model (rule disabled) | ~0.48–0.50 | 99.8% class-1, model defaults to predicting 1, F1-0 collapses near 0 — same statistical trap as `B_has_pos`'s misleading "97.7% accuracy" |

User explicitly authorized the rule after the compliance tradeoff was surfaced (see
`CLAUDE.md`). This session.

## `B_has_pos` / `B_no_pos` — body seen before (10.6% of test combined)

```mermaid
graph LR
  hard["hard rule<br/>(always predict 0 if a positive title exists)<br/>combined Macro F1 0.7998"]
  ret1["title-retrieval + entity-conflict<br/>(word/char TF-IDF sim + entity/number/negation<br/>conflict vs nearest pos/neg candidate)<br/>combined 0.9875 -- THRESHOLD LEAK, see below"]
  honest["same features, HONEST repeated CV<br/>(fit/threshold/eval strictly separated)<br/>combined Macro F1 0.9777"]
  bnopos_leak["B_no_pos only, in-sample threshold<br/>0.9167 -- SAME LEAK, retracted"]
  bnopos_honest["B_no_pos only, honest repeated CV<br/>max_char similarity alone: 0.5755"]
  bnopos_body["B_no_pos: headline-vs-own-body<br/>contradiction features, LogReg<br/>0.5186-0.5422 -- WORSE than retrieval"]
  hard-->ret1-->honest
  ret1-->bnopos_leak-->bnopos_honest-->bnopos_body
```

**Key finding: `B_has_pos` (1278/2011 of the pool) has an almost-deterministic signal
(entity present in title, absent from the single best-matching historical title = swap
signature) and drives the combined number to 0.98 even though `B_no_pos` alone caps at
0.58** — the pool is dominated by the easy regime. Reported separately, not just combined,
because 0.98 hides that `B_no_pos` is still weak.

**Methodology bug found and fixed here:** the first `B_no_pos` result (0.9167) picked its
decision threshold by sweeping directly against the same labels being scored — i.e. no
train/threshold/eval split. With only 34 true negatives in the pool that inflates the
score dramatically. Fixed by nested splitting (fit → separate threshold-selection split →
held-out eval split), repeated 5×5. The honest number (0.575) is far lower and is the one
that should be trusted and reused.

`B_no_pos` structured-feature attempts (headline-vs-body entity/number/negation/action
conflict, LogReg) scored *worse* than plain retrieval similarity — negative result, listed
so it isn't retried.

## `A_cold` — body never seen (63.6% of test, the real bottleneck)

### Baseline and feature-family ablation

```mermaid
graph LR
  v12base["v12: 41 lexical/entity/salience<br/>+ 7 structural chunk-matching feats<br/>LightGBM, dense OOF 0.6631 (Kaggle)"]
  chunkdist["+ chunk-distribution<br/>(quantiles/topk/entropy/run-length)<br/>0.6668 macro_all -- WORSE, reverted"]
  supconf["+ support/conflict/margin<br/>aggregate scores<br/>0.6694 macro_all -- WORSE, reverted"]
  nmftopic["+ NMF topic cosine + KL/JS<br/>divergence<br/>0.6682 macro_all -- WORSE, reverted"]
  action_iso["+ action_conflict alone<br/>(antonym lexicon)<br/>-0.0049 macro -- WORSE, reverted"]
  number_iso["+ number_refined alone<br/>(continuous magnitude gap)<br/>-0.0061 macro -- WORSE, reverted"]
  entrole["+ entity_role_conflict alone<br/>(noisy pool, any-alternative-entity)<br/>+/-0.0000 macro (F1-0 up, F1-1 down, net flat)"]
  v12base-->chunkdist
  v12base-->supconf
  v12base-->nmftopic
  v12base-->action_iso
  v12base-->number_iso
  v12base-->entrole
```

Every single-feature-family addition to the true 48-feature baseline (isolated, properly
ablated) was flat or negative. This is the evidence base for "plateau" — but see below,
the plateau turned out to be about detector *precision*, not about there being no signal.

### Model-capacity experiments (all negative/flat)

| experiment | result |
|---|---|
| LightGBM hyperparameter sweep (200→6000 trees, depth/leaves/lr grid) | flat: 0.6368–0.6450 range, smallest config won — **not a capacity-limited problem** |
| V14: from-scratch hierarchical BiGRU + headline-guided attention + pairwise interaction, PyTorch, weighted BCE | **macro 0.5178, F1-0 0.0934 — far worse than v12.** Confirms an earlier from-scratch neural finding (CNN/Transformer, ~0.53 F1, memorization not generalization) with a *different, larger* architecture — same data-starvation conclusion, not an architecture problem. |
| V14 + v12 ensemble (any blend weight) | monotonic: best weight = 100% v12, 0% V14. No complementary value at any ratio. |
| Paragraph-level MIL (bag=article, instance=~80-word chunk, aggregate MAX/TOP-k/P90) | **all worse than whole-body** (0.599–0.620 vs 0.656) — training-label noise (only ~1 of 4.4 chunks per article actually carries the tamper signal) outweighs the finer granularity |
| Difficulty-aware majority (class-1) undersampling, keep_frac 1.0→0.3 | **monotonically worse** the more dropped (0.6533→0.4972) — with only 483 negatives total, removing majority examples starves the boundary, doesn't sharpen it; literature result doesn't transfer to this small-sample regime |
| Entity co-occurrence graph (1-hop local window / 2-hop corpus-level neighbor) | **worse** (0.6506 vs 0.6561) — signal too sparse (75% of rows have zero 2-hop hits) |

### What worked: model diversity, not new information

```mermaid
graph LR
  v12base2["v12 baseline, 48 feats<br/>LightGBM 0.6422"]
  cat["same feats, CatBoost<br/>0.6503 (beats LightGBM alone)"]
  ens["blend: 0*LGB + 0.5*Cat<br/>+ 0.25*XGB + 0.25*RF<br/>0.6571"]
  bm25s["+ BM25-sentence retrieval feats<br/>(max/top3/top5/std/first-rank)<br/>+ class_weight sweep -> 0:4:1<br/>0.6586, F1-0 0.3540 (best so far)"]
  v12base2-->cat-->ens-->bm25s
```

Diversifying **algorithm** (LightGBM/CatBoost/XGBoost/RandomForest, identical input
features) beat every attempt to diversify **features**. +0.0149 Macro F1 from ensembling
alone — the single largest genuine gain in the whole `A_cold` search.

### Diagnosis correction (this session, human-audited) — see `CLAUDE.md`

Manual read of 100 random cold negatives (not detector output, not top-confidence FN)
found true entity-substitution ≈50% (not 37.8%), a new **subject/object role-reversal**
category, and a concrete numeric-parsing bug ("500 000" invisible to the mismatch regex).
Oracle ceiling for a perfect-precision detector on the obvious-conflict union: **0.62
F1-0** vs the real model's ~0.35 — a precision gap, not proof of an information ceiling.

### Claim Conflict Layer v2 (this session, precision-first typed redesign)

| channel | standalone precision-0 | standalone recall-0 | added to baseline, Δ Macro F1 |
|---|---|---|---|
| entity_role_v2 (typed LOCATION/ORG gazetteer, same-type-only substitution) | 0.1786 | 0.4107 | **+0.0035** (best single addition) |
| number_conflict_v2 (fixes the "500 000" bug, magnitude-tolerant) | 0.0535 | 0.0329 | +0.0029 |
| negation_local_v2 (window around best-matching chunk, not document-wide) | 0.0755 | 0.3101 | −0.0030 |
| all combined | — | — | +0.0024 (precision 0.32→0.42, recall 0.39→0.30 — net neutral) |

Typed entities improved standalone precision only marginally (0.15→0.18) despite a
proper LOCATION/ORG gazetteer replacing the old capitalized-word-only pool. **The
precision-first hypothesis is confirmed directionally but the gap to the 0.62 oracle
ceiling is still wide open** — most flagged "conflicts" are still coincidental
co-occurrence, not genuine substitution. Current best config unchanged:
**48 feats + BM25-sentence + class_weight 0:4:1 + entity_role_v2, Macro F1 ≈ 0.6596,
F1-0 ≈ 0.358.**

## Combined-pipeline estimate

Weighted-confusion-matrix combination (not a naive average — see method note in session)
of `C_exact` (rule, 1.0) + `B` (title-retrieval, 0.9777) + `A_cold` (ensemble, ~0.66) at
their real test-share weights (25.7% / 10.6% / 63.6%) estimates **Macro F1 ≈ 0.857** for
the fully compliant pipeline. This has not been run end-to-end and submitted for real
confirmation as of this commit — see `SUBMISSIONS.md` for what has actually been
submitted.
