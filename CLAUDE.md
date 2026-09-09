# Penyisihan IFEST 2026 DAC — Working Agreement

<!-- BRIEF:START -->
## Domain brief

**Task.** Binary classification: is an Indonesian news headline consistent with its
article body. 14,397 train / 3,603 test, ~90/10 imbalanced (class 1 = consistent).
**Metric: Macro F1.**

**Rules.** No pretrained models of any kind (IndoBERT, BERT, **XLM-R** named explicitly,
FastText-pretrained, LLM, VLM, API models). Allowed: TF-IDF, CountVectorizer, BM25, and
embeddings trained from scratch on the competition corpus only. Max 3 submissions/day, 2
final submissions. **v2–v6 all used IndoBERT/XLM-R and are disqualifiable** — discovered
only after 5 submissions; every notebook from v8 onward is a from-scratch rebuild.

**The corpus has zero punctuation.** Verified directly on raw `content`: 0 periods, 0
newlines, in every row sampled. There are no sentences and no paragraphs to split on —
every "sentence-level" or "paragraph-level" idea in this project had to fall back to
fixed-width word-chunks (50-word windows, 25-word stride for BM25; ~80-word non-overlap
chunks tried for a paragraph-MIL experiment). Any future plan that assumes real sentence
boundaries will not survive contact with this data — check the raw field first.

**The negative-generation mechanism (verified on real data).** One true headline per
article; negatives are the same body paired with a *tampered* headline. Of 1,097 same-body
title pairs, 529 differ by exactly one token, 95% of swapped tokens are proper nouns, 96%
of substitutes come from the corpus's own entity pool. **0 exact (title, content) pairs
ever carry conflicting labels — no label noise.** Mutation census over 468 same-body
minimal pairs: entity 92.3%, number 5.1%, generic 1.9%, action 0.6%, negation 0%, date 0%.

**Test decomposes into 4 regimes, and each needs a different method — this is the single
most important structural fact in the project:**

| regime | share of test | mechanism | best method found | Macro F1 |
|---|---|---|---|---|
| `C_exact` — exact (title,body) pair already in train | 25.7% | none (lookup) | rule: copy the known label | **1.0** (0 label noise, provable) — but see the compliance note below |
| `B_has_pos` — body seen, train already has a label-1 title for it | 6.0% | one title is genuine, rest are tampered | title-retrieval + entity-conflict classifier | **0.996** (honest CV) |
| `B_no_pos` — body seen, no known-positive title yet | 4.6% | ambiguous which titles (if any) are genuine | retrieval-only, `max_char` similarity | **0.575** (honest CV; ceiling, not a bug — see below) |
| `A_cold` — body never seen before | **63.6%** | genuine generalization, no memory to exploit | ensemble (CatBoost+XGB+RF) on 48 handcrafted features | **0.66** (honest grouped CV) — the real bottleneck |

`A_cold` is 63.6% of test and by far the weakest regime — it dominates the overall score
far more than its intrinsic difficulty alone would suggest, because everything else is
close to solved.

**Compliance judgment call — the `C_exact` rule.** The rules also say *"dilarang ...
merekonstruksi label"* (label reconstruction is forbidden), which is ambiguous for a raw
label-copy on an exact-duplicate pair. This project's rule-ablation measured the ambiguous
component (`C_exact` copy) as worth only −0.0016 Macro F1 if removed, versus `B_has_pos`
(probabilistic inference from construction, not a copy) at −0.3439 — so the two “rule”
components are not equally risky. **The user explicitly authorized using the `C_exact`
rule** after this tradeoff was surfaced (2026-09-09); this is recorded here, not decided
unilaterally. If in doubt, ask the organizers before relying on it in a real submission.

**`B_no_pos`'s 0.575 ceiling is real, not a bug — measured from two independent angles.**
(1) Negative-title cluster coherence is weak and high-variance (mean pairwise similarity
0.42, std 0.37) — negatives for one body come from heterogeneous mechanisms, so "similar to
a known negative" is not a reliable rule. (2) A from-scratch headline-vs-body contradiction
model (entity/number/negation features, Logistic Regression) scored *worse* (0.54) than
retrieval alone. Two independent approaches converge on the same ceiling with only 34
true negatives in the pool — that convergence, not a single failed experiment, is the
evidence this is near the real limit for this regime's sample size.

**`A_cold` diagnosis changed mid-project and this matters for anyone continuing the work.**
Early conclusion: "~73% of false negatives are semantic, might be an information ceiling."
That was wrong in an important way — it conflated *"our automated detector can't explain
this error"* with *"there is no detectable mechanism here."* A 100-example **manual**
audit (not detector output) found the true entity-substitution rate is closer to **~50%**
of cold negatives (not the 37.8% our automated pool-based detector reported), plus a
previously-unbuilt category — **subject/object role reversal** ("pasien kaget lihat dokter"
vs the reverse) — and confirmed a concrete regex bug: numbers written with space-separated
thousands ("500 000") were invisible to the mismatch detector. The oracle ceiling for a
*perfect-precision* detector on the obvious-conflict union is **0.62 F1-0** against a real
model at ~0.35 — meaning the gap is a **precision problem in feature engineering**, not
proof of an information ceiling. A typed-entity redesign (Claim Conflict Layer v2, this
session) narrowed but did not close that gap — standalone entity-role precision only moved
0.15→0.18. **The honest state as of this session: real, structured signal is
under-exploited, but nobody has yet found the feature design that captures it at usable
precision.** Don't restate "A_cold is fundamentally semantic" as settled fact; it isn't.

**Compute policy.** Local box is CPU-only (a laptop, no GPU). Prefer local dry-runs for
anything under ~5 minutes; anything neural or GPU-shaped goes to Kaggle T4
(`--accelerator NvidiaTeslaT4` on `kernels push`, `enable_gpu: true` risks a P100 that
recent torch wheels can't compile for). **Always dry-run locally before pushing to
Kaggle** — the sentence-splitting bug (this corpus has no punctuation) was caught this way
before it wasted a Kaggle run.

**Oracle-boundary discipline.** Never use the Kaggle leaderboard score itself (public or a
suspicious unexplained submission) to select features, thresholds, or model choices — that
is leaderboard hillclimbing, the same failure mode flagged in this user's other projects.
Every comparison in `EXPERIMENTS.md` is from local/Kaggle **cross-validation**, not from
watching the submission score change. Threshold selection must be done on a held-out split
disjoint from the split being scored (an early bug here — picking a threshold on the same
OOF array being evaluated — silently inflated a `B_no_pos` result from a real ~0.575 to a
fake-looking 0.9167; see `EXPERIMENTS.md`).
<!-- BRIEF:END -->
