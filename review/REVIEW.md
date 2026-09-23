# Detailed review: every problem found

External audit of `main` at commit `99ff91a` (2026-09-19), done 2026-09-23. It merges two independent reviews: this audit, and a second reviewer's report the team shared. Where the two disagree, the entry says which claim the data supports.

Each finding has an ID so you can reference it in issues and PRs. Severity:

- **Critical**: changes a headline result or would get the paper rejected on its own.
- **Major**: a reviewer will flag it; fix before submitting.
- **Minor**: polish.

Evidence status:

- **[data]**: recomputed from the repo's shipped `results/`. `python3 review/audit.py` reproduces it.
- **[code]**: read directly in the source.
- **[web]**: checked against the live paper or policy page.

Line links point at `main`.

## Contents
- [What holds up](#what-holds-up)
- [V: Validity of the experiments](#v-validity-of-the-experiments)
- [S: Statistics and inference](#s-statistics-and-inference)
- [R: Reproducibility and provenance](#r-reproducibility-and-provenance)
- [C: Citation case study](#c-citation-case-study)
- [N: Novelty and attribution](#n-novelty-and-attribution)
- [X: Scope and claims](#x-scope-and-claims)
- [T: Text that contradicts the data or code](#t-text-that-contradicts-the-data-or-code)
- [P: Submission compliance (IEEE Access)](#p-submission-compliance-ieee-access)
- [Where the two reviews differ](#where-the-two-reviews-differ)

---

## What holds up

Both reviews agree on these. Keep them.

1. **The analysis reproduces.** From `results/raw/`, the pipeline regenerates `numbers.tex` and all three tables byte-for-byte. The only exception is `\nGenerations`: it counts rows in an unshipped cache, so it prints 0. [data]
2. **The estimator tests pass** as claimed: pass@k against brute force, bootstrap coverage 0.963, paired-bootstrap FPR 0.057. [code]
3. **Selection efficiency η = (A − p1)/(C − p1)** is an interpretable normalization: 0 is a random pick, 1 is the oracle. The affine form is not foregrounded in the closest prior work (see N2). It is small but usable.
4. **"Coverage headroom vs. ability to select from it"** is useful engineering language.
5. **Cheap calibration before building an expensive system** is the right practical question.
6. **The counterbalanced pairwise probe** (both orders) is a good idea. It needs the fixes in S3.
7. **Task-level paired bootstrap plus Holm correction** is better statistics than most agent papers use.
8. **The limitations are unusually candid.** Tiny models, synthetic tasks, Gaussian violations and judge-only R² are all stated.
9. **The code is clean**: cached generation, per-call token accounting, all numbers generated as macros.

There is a paper in here. It is narrower than the current title and abstract.

---

## V: Validity of the experiments

### V1 (Critical): the token-matched headline is mostly a truncation artifact [data][code]
**Setup.** Ensemble members are capped at 320 completion tokens on CSP and GSM8K ([`run.py#L43`](../gvgap/run.py#L43)). The long-CoT arm that defines the single-agent frontier gets 2× and 4× that cap ([`run.py#L40`](../gvgap/run.py#L40)).

| cell | pool samples hitting the cap | acc. if truncated | acc. if complete |
|---|---|---|---|
| 0.5B CSP | 92% | 0.027 | 0.344 |
| 1.5B CSP | 78% | 0.043 | 0.596 |
| 0.5B GSM8K | 35% | 0.060 | 0.423 |
| 1.5B GSM8K | 11% | 0.114 | 0.729 |

**Impact.**
- 12 of the 13 Holm-significant negative "matched lifts" are CSP cells.
- Drop CSP and the median matched lift goes from +0.047 (naive) to +0.006 (matched). The paper reports a sign flip, +0.036 → −0.028; without CSP there is none.
- The README showcase ("0.644 for 701 tokens vs. 0.478 for 2,947 with an oracle") compares a 1,280-token cap to a 320-token cap.

**Fix.** Use one cap for every arm, set high enough that fewer than 1% of samples truncate. Log the truncation rate per cell and fail the run above the threshold.

### V2 (Critical): the baseline gets a better prompt than the ensembles [data][code]
**Setup.** The appendix says all arms share one system prompt, "so that differences are attributable to orchestration rather than prompt engineering". But `seq_long` appends "Think through this carefully and at length… check your working" ([`arch.py#L409`](../gvgap/arch.py#L409), [`prompts.py#L68`](../gvgap/prompts.py#L68)). Ensemble members never get that instruction.
- On Logic (1.5B), the base prompt yields about 21 completion tokens with accuracy 0.48.
- The long-CoT prompt yields about 200 tokens with accuracy 0.59.

**Impact.** Section 3 defines the frontier as sequential refinement only. Under that definition:
- 24% of apparent gains flip, not 47%.
- The median matched lift is +0.017, so there is no sign flip.
- 33% of cells are net-negative, not 56%.

**Fix.** Give every arm the same base prompt. If long CoT is a baseline, also build ensembles from long-CoT samples; that is what self-consistency is in Wang et al. 2023.

### V3 (Major): the frontier is an in-sample maximum over seven noisy arms [data]
**Setup.** `upper_envelope` ([`analysis.py#L149`](../gvgap/analysis.py#L149)) takes the running max of single, refine×4 and long×2 on the same tasks it is then compared against. The code calls this "conservative".

**Impact.** It biases the baseline upward. Choosing the arm on calibration and scoring it on test shows +0.022 optimism on average, and up to +0.10 in one cell. That pushes every matched lift down.

**Fix.** Pre-specify the baseline, or choose it on calibration and evaluate on test.

### V4 (Major): "compute-matched" is often not matched; the baseline is clamped [data][code]
**Setup.** When a multi-agent cell costs more than the largest measured single-agent budget, `interp_accuracy` returns the endpoint accuracy ([`metrics.py#L134`](../gvgap/metrics.py#L134)). The single agent is never run at that budget.

**Impact.** 41 of 64 cells are priced beyond the largest measured single-agent budget: median 1.34×, up to 2.37×. In those cells the multi-agent system spends more than the baseline it is compared to. This biases toward multi-agent, so it does not manufacture the "multi-agent loses" result. It does make "compute-matched" in the title inaccurate. (Found by the second reviewer; quantified here.)

**Fix.** Run the single agent at every budget used: more refine rounds, 8× and 16× long-CoT caps. Otherwise call it "token-budget controlled" and report the clamped cells separately.

### V5 (Major): tokens are a weak proxy for compute [code]
**Setup.**
- Parallel samples pay their prompt k times ([`run.py`](../gvgap/run.py) sums `total_tokens` per sample). With prefix caching, a deployed system pays the shared prompt once.
- Attention cost, latency and $ cost do not track total tokens across these architectures.

**Fix.** Report at least four cost models: total tokens, completion tokens, prefix-cached tokens, and wall-clock/GPU-seconds (or $). Show which conclusions survive all of them. See roadmap angle 2.

### V6 (Major): the judge, the G probe and debate see truncated candidates [data][code]
**Setup.** The judge and probe see `text[:700]` of each candidate ([`arch.py#L156`](../gvgap/arch.py#L156), [`arch.py#L198`](../gvgap/arch.py#L198)). Debate agents see `text[:500]` of their peers ([`arch.py#L317`](../gvgap/arch.py#L317)). Final answers come last.

**Evidence.** I replicated the probe's pair sampling (`Random(11)`); pair counts match the paper in all 12 cells. In GSM8K calibration pairs, a candidate exceeds roughly 175–230 tokens (≈700 chars) in:
- 12–17 of 20 pairs (1.5B);
- 17 of 18 pairs (0.5B).

**Impact.** "LLM judges gain nowhere" and "G ≈ 0 on GSM8K" partly measure an instrument that cannot see the answer.

**Fix.** Pass full texts. If context forces a cut, trim the middle and keep the final answer.

### V7 (Major): the 0.5B "judge" is a position picker [data]
**Evidence.** Pick-A = 1.00 in 4 families and 0.00 in WordCon. Order consistency is 0.00 everywhere except MBPP (0.20).

**Impact.** G ≡ 0 by construction for all six 0.5B cells, so half of the "scale" axis carries no verification signal.

**Fix.** Treat the 0.5B model as a format-compliance failure. Report it as such and drop it from G→η validation.

### V8 (Minor): task construction [code]
- Logic asks "Who is the taller?" over 4–6 people ([`synth.py#L113`](../gvgap/synth.py#L113)). It should say "tallest".
- MBPP is graded on all tests, including the one shown in the prompt. The paper says "held-out remainder" ([`verify.py#L145`](../gvgap/verify.py#L145)).

---

## S: Statistics and inference

### S1 (Critical): R² = 0.84 is not evidence for the G theory [data][code]
**Setup.** The 24 prediction cells use three different predictors ([`predict.py`](../gvgap/predict.py)):

| arm | how the prediction is made | cells | R² within the arm |
|---|---|---|---|
| program verifier | η hard-coded to 1 ([`L128`](../gvgap/predict.py#L128)), so prediction = calibration headroom. On CSP/WordCon the verifier is the grader | 6 | 0.53 |
| self-consistency | "plurality alignment" = the self-consistency lift itself, measured on calibration ([`L123`](../gvgap/predict.py#L123)) | 8 | 0.23 |
| LLM judge | G → η via the latent-score model: the only arm that tests the theory | 10 (5 are 0.5B with G ≡ 0) | 0.52 |

**Impact.**
- Predict every judge cell as 0, ignoring G entirely, and overall R² is still **0.818**. G adds about 0.02.
- The paper calls transfer R² = 0.80 an "upper bound" on any diagnostic, but its own a priori R² (0.84) exceeds it.

**Fix.** Report the judge arm as the test of the theory, with baselines. Drop the pooled R² headline.

### S2 (Critical): G is statistically indistinguishable from zero everywhere [data]
**Evidence.** G is estimated from 5–20 pairs per cell. With 95% Clopper–Pearson intervals over pairs, **every calibration G interval includes 0**. The largest, +0.42 (71% pairwise accuracy), has CI [−0.30, +0.80]. The paper calls this range "near-perfect discrimination".

**Fix.** Aim for at least 100 pairs per cell: several pairs per task, several tasks. Report CIs everywhere G appears.

### S3 (Major): counterbalancing breaks when the judge abstains [data][code]
**Setup.** Unparsed verdicts drop one of a pair's two presentations ([`arch.py#L244`](../gvgap/arch.py#L244)), so a position picker no longer scores exactly chance.

**Evidence.** 1.5B ARC:
- Calibration: order consistency 0.00, abstention 21%, yet G = +0.09.
- Test: pick-A 0.00, yet G = +0.08.

**Fix.** Drop both presentations of a pair when either abstains, and report the abstention rate.

### S4 (Major): the build/don't-build rule is not held out [code]
**Setup.** The prediction uses calibration data. The "actual" outcome is read from `matched_lift_ci.csv` ([`predict.py#L198`](../gvgap/predict.py#L198)), which is computed on all 90 tasks, including those 30 calibration tasks ([`analysis.py#L235`](../gvgap/analysis.py#L235)).

**Evidence.** The result itself is modest: 26 cells, 73.1% agreement vs. a 61.5% always-"don't build" baseline, precision 0.636, recall 0.70. (Second reviewer; verified.)

**Fix.** Compute outcomes on the test split only, or drop the claim.

### S5 (Major): WordCon duplicates leak across splits and inflate n [data]
**Evidence.**
- 90 WordCon tasks contain only 42 unique prompts. 25 of 60 "held-out" tasks are verbatim calibration prompts.
- For 0.5B, all 24 duplicate groups have byte-identical 8-sample pools (served from cache). 15 of those groups span calibration and test, so identical data sits in both splits.
- The bootstrap treats these as independent tasks.

**Fix.** Deduplicate in `build_data.py` and assert that splits are disjoint.

### S6 (Minor): smaller inference issues [code][data]
- Token-matched CIs use `n_boot = 2000` ([`analysis.py#L224`](../gvgap/analysis.py#L224)). The paper says 10,000.
- Tables 1 and 3 are 1.5B-only, and the captions do not say so. Table 3 shows whichever of k=4 or k=8 has the larger |lift| per cell ([`report.py#L148`](../gvgap/report.py#L148)).
- Plurality alignment computes p1 over all tasks but mode accuracy only over tasks with a parsed answer ([`arch.py#L381`](../gvgap/arch.py#L381)).

---

## R: Reproducibility and provenance

### R1 (Major): the README's reproduction commands do not reproduce the paper [code]
**Setup.**
- `build_data.py` writes 120 tasks per family (30 calibration / 90 test; [`L23`](../gvgap/build_data.py#L23)). `run.py` runs all of them unless `GVGAP_LIMIT=90` is set ([`L51`](../gvgap/run.py#L51)). That variable is documented nowhere.
- The shipped results (90 tasks, 30/60) are consistent with the current code **plus** `GVGAP_LIMIT=90`.

**Fix.** Make 90 the default or document it, and write the task count into the results.

### R2 (Major): the paper describes a different configuration than the code runs [code][data]
| paper | code |
|---|---|
| k ∈ {2,4,8}, three sub-pools | `K_VALUES = [4, 8]`, `SUBSETS = 2` ([`run.py#L37`](../gvgap/run.py#L37)) |
| debate over 2 or 3 rounds | `DEBATE_CFGS = [(3, 2)]` only ([`run.py#L41`](../gvgap/run.py#L41)) |

The committed raw results contain `k4`/`k8`, `n_sub = 2` and `debate_a3r2`. They **match the code**; the paper text is what is out of date (see [the differences section](#where-the-two-reviews-differ)).

### R3 (Major): a cache-key collision makes results depend on cache state [data][code]
**Evidence.** In 1.5B WordCon, `seq_r1` ≠ `single` on 40 tasks: accuracy 0.0889 vs 0.1000, tokens 77.82 vs 77.92. The second reviewer found this. They should be identical, because both are the first sample of the same prompt.

**Cause.** All 40 are duplicated prompts. The cache key is (model, prompt, max_tokens, temp, seed) ([`engine.py#L108`](../gvgap/engine.py#L108)):
1. Duplicates generated in the same batch get different samples.
2. `INSERT OR REPLACE` keeps only one ([`engine.py#L124`](../gvgap/engine.py#L124)).
3. Later arms read that one.

This is also why the 0.5B duplicates are byte-identical (S5).

**Impact.** Re-running from cache changes the 1.5B WordCon pools.

**Fix.** Key the cache by item ID and sample index as well.

### R4 (Major): no provenance, and stale results are silently reused [code]
- `run.py` skips any family whose output file exists ([`L216`](../gvgap/run.py#L216)). There is no hash of code, config, data or model revision. (Second reviewer; verified.)
- The seed depends on batch offset: `mx.random.seed(seed*1_000_003 + start)` ([`engine.py#L158`](../gvgap/engine.py#L158)). Samples therefore depend on what was already cached.
- The model revision (HF commit SHA) and the mlx/mlx-lm versions are not recorded. `requirements.txt` uses `>=` pins.

**Fix.** Write git SHA, config hash, data hash, model revision and package versions into every raw JSON. Refuse to skip when they differ. Pin versions.

### R5 (Major): generation texts are not shipped [code]
**Setup.** The generation cache is gitignored.

**Impact.**
- Nobody can re-grade outputs, audit answer extraction (`extraction_audit.py` exists but its output is not shipped), check the "21,847 generations" count, or measure the judge-truncation issue (V6) directly.

**Fix.** Release all generations as JSONL, as a GitHub release asset, Zenodo record or HF dataset.

### R6 (Minor): small artifact gaps [code][data]
- `refs.bib` says every entry was checked by `gvgap/verify_refs.py`. That file is in no commit.
- `matched_lift_ci.csv` is not bit-reproducible here: CI bounds differ by up to 0.011 and Holm p by up to 0.18. The same 22 cells are significant. It is likely a library-version difference; pin versions.
- `nArch` is hard-coded to 8 ([`report.py#L191`](../gvgap/report.py#L191)).

---

## C: Citation case study

### C1 (Critical): registry η = 1.00 is true by definition [code]
**Setup.**
- Ground truth is "a registry title matches at ≥ 0.88".
- The "registry selector" `select_oracle_api` ([`run_citations.py#L152`](../gvgap/run_citations.py#L152)) returns the first candidate whose ground truth is true.

So η = 1 is an identity, yet it appears in the abstract and cover letter as a finding. The paper does admit that the registry "supplies both ground truth and one of the two selectors".

### C2 (Major): truth is a title match only [code]
**Setup.** Matching uses `SequenceMatcher` on titles ([`citations.py#L217`](../gvgap/citations.py#L217)). Author and year are generated but never checked. It is not a paper-identity oracle, and "near-perfect verifier" overstates it.

**Evidence.** The cache that feeds ground truth has 7 entries at ≥ 0.88:
- The junk "title" "Smith et al" matches "Smith, et al" at 1.00.
- "Handwritten character recognition using deep belief networks" matches a different paper ("Bangla handwritten…").
- "Surface codes for fault tolerant quantum computation" matches "Holonomic surface codes…".

Pools are not saved, so which entries entered the final run cannot be checked.

### C3 (Major): G = −0.67 rests on 6 pairs and is not counterbalanced [data][code]
**Evidence.** Only 6 of 40 topics contain any "real" candidate, so G = −0.67 is 1 hit in 6 pairs. The side is randomized once per pair ([`run_citations.py#L177`](../gvgap/run_citations.py#L177)). That is the method the paper itself rejects in R2, where it produced a fake G = +0.40.

The paper describes this as "every discriminating pair available" resting on "far more data".

### C4 (Minor): text vs. code [code][data]
- The paper says OpenAlex. The run used CrossRef: 105 of 105 calls; the cache holds 242 CrossRef and 31 OpenAlex entries. The commit message says "CrossRef with OpenAlex fallback".
- "Correctly anticipating that this critic is worse than useless": the prediction was η = −0.18; the observed value is 0.00.

---

## N: Novelty and attribution

Every item below is uncited in the paper unless noted. Full list with links and BibTeX keys: [PRIOR_WORK.md](PRIOR_WORK.md) and [`prior_work.bib`](prior_work.bib).

### N1 (Critical): the name and the thesis already exist [web]
- The paper says "we introduce the generator–verifier gap".
- Noam Brown gave a talk titled **"The Generator-Verifier Gap"** (Simons Institute, Sep 2024): when verifying is easier than generating, extra inference compute pays.
- Song et al. (ICLR 2025) formalized a **"generation-verification gap"** for LLMs. Their definition differs from G = 2q − 1, so this is not copying, but a 2026 paper cannot skip it.
- Wei (2025) restated the thesis as "asymmetry of verification / verifier's law".
- Weaver (NeurIPS 2025) is literally titled *Shrinking the Generation-Verification Gap by Scaling Compute for Verification*.

### N2 (Critical): the decomposition has close precedents [web]
- **PRISM / DIANOIA** (Yang et al., EMNLP 2026; arXiv Feb 2026; v1 was titled PRISM):
  - It states E[Q] = C_K · η with η called "effective efficiency".
  - It says "a high baseline p leaves small exploration headroom C_K − p".
  - It says a deterministic verifier lifts selection to its ceiling.
  - It uses GSM8K and MBPP with budget-matched comparisons.
- **Hu (Jul 2026)**, *Oracle Gap and Signal Fidelity*: selector gains on a fixed pool are bounded "first by the oracle gap and then by signal fidelity", as a pre-deployment diagnostic.
- Saunders et al. (2022, GDC gaps vs. oracle best-of-2) and weak-to-strong "PGR", which has η's affine form (Burns 2023 is in `refs.bib` but never cited).

The affine normalization (0 random, 1 oracle) is not foregrounded in these. Claim it as a presentational contribution, not a theorem.

### N3 (Major): G is a known quantity [web]
- q is Chen et al.'s p_comp (NeurIPS 2025).
- For a scoring verifier q is the AUC, so G = 2·AUC − 1 (Somers' D).
- Under the paper's own i.i.d. sampling, **G equals η at k = 2**: per task, A(2) − p1 = p(1−p)(2q−1) and C(2) − p1 = p(1−p).

### N4 (Major): Proposition 2 is textbook [web]
The latent-score model is **Thurstone–Mosteller / equal-variance signal detection theory**:
- q = Φ(d′/√2) is the 2AFC formula.
- P(1,k,δ) is the mAFC integral (Green & Swets 1966; Hacker & Ratcliff 1979; Macmillan & Creelman 2005; DeCarlo 2012).
- P(c,k,δ) also has a closed form, so Monte Carlo is not needed.
- The equal-variance assumption is contradicted for LLMs by Cacioli (2026): z-ROC slopes 0.52–0.84.

What is new is **applying** 2AFC→mAFC to predict an LLM judge's best-of-k accuracy. That is the one piece worth building on.

### N5 (Major): plurality alignment is known [web]
- It is the plurality Condorcet jury theorem (List & Goodin 2001), Wu et al. ICLR 2025 Thm 1, and Chen et al. NeurIPS 2024 (d_V, easy/hard queries).
- In the code it is literally the self-consistency lift.

### N6 (Critical): compute-matched multi-agent evaluation is established [web]
- **Tran & Kiela (Apr 2026)**: single agents match or beat five multi-agent designs at equal thinking-token budgets.
- **Mirzaei (Jul 2026)**, *Sample More, Reflect Less*: 1.5B–7B models; debate, self-picked best-of-N, Self-Refine and Reflexion all compared to repeated sampling at equal measured tokens, with paired bootstrap and multiplicity correction. Finds "no method is reliably better than repeated sampling at equal cost". This is the closest empirical precedent.
- **Kim et al.**, *Towards a Science of Scaling Agent Systems*:
  - v1 (Dec 2025): 180 configurations, CV R² 0.513, picks the best architecture for 87% of held-out configurations.
  - v3 (Apr 2026): 260 configurations, CV R² 0.373–0.413.
  - It builds a predictive model from measurable properties under standardized budgets.
- **Hu, Shen & Lakshmipathi (May 2026)**: tried to predict when debate helps at a matched budget and failed.
- **Earlier work**:
  - Singhi et al. (COLM 2025): at matched compute, self-consistency beats generative verifiers.
  - Choi et al. (NeurIPS 2025): majority voting explains most debate gains.
  - Smit et al. (ICML 2024); Wang et al. (ACL 2024); Wang et al. (EMNLP 2024, *Reasoning in Token Economies*).
  - Huang et al. (ICLR 2024, Table 7): debate loses to self-consistency with the same number of responses. **The paper cites this one**, but for something else.

### N7 (Major): best-of-N and verifier work [web]
- Huang et al. (ICML 2025), *Is Best-of-N the Best of Them?*: coverage and imperfect reward models.
- PairJudge RM (2025): pairwise judges for best-of-N.
- Generative Verifiers (ICLR 2025).
- Lu et al. (2025), *When Does Verification Pay Off?*: a "verifier gain" that predicts rejection-sampling gains, across 37 models including Qwen2.5-0.5B/1.5B.

### N8 (Critical): the framing claims are false as of 2026 [web]
Both of these are contradicted by N6/N7:
- "Compute-matched comparison is still not standard practice".
- "There is no measurable quantity that says when to expect a gain" / "existing accounts are largely taxonomic".

**Fix.** Make a narrower claim. The bibliography cites none of the key papers above.

### N9 (note): plagiarism
- **Text**: nine distinctive sentences appear nowhere online except this repo. Against the 8 closest sources there are no shared 6-word runs. This is a web search, not iThenticate over the full scholarly corpus.
- **Ideas**: the uncited ideas above (N1–N8) are an **attribution failure**. IEEE's plagiarism definition covers reusing "prior ideas … without explicitly acknowledging the original author". Fix the attribution; nothing here suggests copying.

---

## X: Scope and claims

### X1 (Critical): the models are too small for the conclusions
- Qwen2.5-0.5B and 1.5B are small and old for September 2026. One of them cannot do pairwise judging at all (V7).
- The limitations section admits this, but the abstract and conclusion still generalize: "ensembles selected by the model judging itself gain nowhere"; "the question is well-posed and cheaply decidable".

### X2 (Major): synthetic families with exact checkers are sanity checks
On CSP and WordCon the "program verifier" is the grader. `check_wordcon` is literally both. η = 1 there is by construction; the paper admits this. It demonstrates the ceiling, not a finding.

### X3 (Major): the claims are broader than the architectures tested
- Only homogeneous agents of one model are tested: no heterogeneous agents, tools, decomposition or role specialization.
- The conclusion characterizes what multi-agent structure "is".

---

## T: Text that contradicts the data or code

| where | text says | data or code says |
|---|---|---|
| Intro | η ∈ [0,1] | Prop. 1: η < 0 attainable; 5 cells are negative |
| R4 | "Debate is the most expensive arm" | The LLM judge costs more in all 6 families (CSP 5,012 vs 2,532 tokens) |
| R6 | "the same family can move from a negative to a positive gap as the model grows" | No family does. 0.5B G = 0 everywhere; WordCon goes 0 → −0.27 |
| R2 | "near-perfect discrimination" | Max G = +0.42 (71%), CI includes 0 |
| R5 | transfer R² "upper-bounds what any diagnostic could achieve" | The a priori R² (0.84) exceeds it (0.80) |
| Setup | k ∈ {2,4,8}, three sub-pools, debate 2 or 3 rounds | k ∈ {4,8}, two sub-pools, 3×2 only |
| Setup | 10,000 resamples | 2,000 for the token-matched CIs |
| Setup | MBPP graded on the held-out remainder | Graded on all tests |
| Sec. 3 | frontier = sequential refinement | refinement + long CoT (as the README says) |
| App. | one prompt for all arms | long CoT adds a reasoning instruction |
| Case study | OpenAlex | CrossRef (105/105) |
| Case study | G from "every discriminating pair", "far more data" | 6 pairs |
| README | Logic judge "ends up *below* a single sample" | 0.478 vs 0.478 |
| Fig. 1 | "same headroom" | +0.20 vs +0.32 |
| Tables 1 and 3 | no model named | 1.5B only; Table 3 shows the larger-magnitude k |

---

## P: Submission compliance (IEEE Access)

Checked on ieeeaccess.ieee.org and the IEEE Author Center, 2026-09-23.

| requirement | status |
|---|---|
| IEEE Access template: "submissions … will be returned to draft or immediately rejected" otherwise | Uses generic `IEEEtran` [journal]. **Blocking** |
| Short biographies for all authors, below the references | Missing. **Blocking** |
| Submitting author has a public, filled-in ORCID | Marked "needed" in `form_fields.md` |
| AI disclosure: AI-generated "text, figures, images, and code" must be disclosed in the acknowledgments, naming the system and the sections | Missing. Three commits carry `Co-Authored-By: Claude Opus 5 (1M context)` (IEEE build, submission package, cover letter), and the figure appendix uses Claude Code's OKLab palette validation. **Blocking** |
| APC | US$2,160 plus tax. The package says "~US$2,000" |
| Preprint | Allowed, with the IEEE notice on the posted version |
| Reviewer criteria | contribution; technical soundness; references "applicable and sufficient"; well-designed study with data sufficient for the conclusions; originality; conclusions supported by the data. This version fails at least three |

Four `refs.bib` entries also drop authors without saying so (Brown, Clark, Huang, Wang). All 23 references are real and correctly identified.

---

## Where the two reviews differ

The second reviewer's findings are all included above. Three conclusions need adjusting:

1. **"The committed results came from a different version of the experiment than the released code."**
   - The committed results match the current code: `k4`/`k8`, `n_sub = 2`, `debate_a3r2`, and 90 tasks via `GVGAP_LIMIT=90`.
   - What is out of date is the paper text (R2), plus an undocumented environment variable (R1).
2. **The `seq_r1` ≠ `single` WordCon mismatch was offered as evidence of that version drift.**
   - It is a cache-key collision on duplicated prompts. All 40 mismatches are on duplicated prompts (R3).
   - It is still a real reproducibility bug.
3. **"PRISM" and "DIANOIA" are the same paper** (arXiv 2602.08586, v1 = PRISM, EMNLP 2026). Kim et al.'s numbers differ by version: v1 has 180 configurations and R² 0.513; v3 has 260 and 0.373–0.413.

Two claims this audit adds and the second review did not have: the truncation confound (V1) and the prompt confound (V2). Together they account for most of the headline sign flip.
