# Roadmap: from here to a publishable paper

Finding IDs (V1, S2, N6, …) refer to [REVIEW.md](REVIEW.md). Bib keys refer to [prior_work.bib](prior_work.bib).

Time and readiness figures below are judgment estimates, not measurements.

## 1. Where you are: about 20% of the way

| Dimension | Now | Why |
|---|---|---|
| Code and analysis infrastructure | ~70% | Clean, cached, tested; numbers regenerate from raw data. Keep it |
| Statistical method | ~50% | Paired bootstrap and Holm are good. G is underpowered (S2), abstentions break counterbalancing (S3), the decision rule leaks (S4), WordCon duplicates (S5) |
| Experimental validity | ~15% | Truncation (V1) and prompt (V2) confounds drive the headline. Clamped baselines (V4). Judges see cut-off answers (V6) |
| Scale and generality of evidence | ~10% | Two small models from one family; half the scale axis can't judge (V7) |
| Novelty and positioning | ~10% | Every headline claim has prior art (N1–N8). One narrow new idea survives |
| Reproducibility and provenance | ~35% | Undocumented env var (R1), cache collisions (R3), no provenance (R4), no generation texts (R5) |
| Claims calibrated to evidence | ~30% | The abstract and conclusion generalize far past two small models (X1) |
| Venue compliance | ~20% | Wrong template, no bios, no AI disclosure (P) |

**What that means.** The reusable asset is the code base and the question. The results must be regenerated, and the contribution must be re-scoped around the one thing that is new (N4): using a cheap pairwise probe plus signal detection theory to forecast listwise LLM-judge selection. Everything else is either known or needs re-measuring.

## 2. Strategy: which paper to write

Don't write "multi-agent structure is not a general accuracy technique". Mirzaei 2026, Tran & Kiela 2026, Kim et al. and others have already published that.

Write the narrower, testable paper. Candidate titles:

- *Forecasting Best-of-N Selector Gains from Pairwise Verifier Calibration under Fixed Token Budgets*
- *A Signal-Detection Model of LLM Judges Predicts Best-of-k Selection*

Core claim to earn: *a counterbalanced pairwise probe on ~30–50 labelled tasks predicts a judge's best-of-k selection efficiency on held-out tasks, and whether that selector beats a measured single-agent baseline at the same budget, across models and task families, better than existing diagnostics.*

Rename G to **pairwise verifier discrimination** (report it as AUC or d′). Cite Brown/Song for the generator–verifier gap instead of claiming to introduce it (N1).

Target **TMLR** first. Its acceptance criteria (checked 2026-09-23) are:
- "Are the claims made in the submission supported by accurate and convincing evidence?"
- Would some of TMLR's audience be interested, and is it communicated clearly?
- It explicitly says "novelty of the studied method is not a necessary criterion".

That plays to this project's strength, careful measurement, and neutralizes its weakness, novelty. A workshop version in parallel gets fast feedback. IEEE Access is a fallback (section 8).

## 3. Phase 0: triage (1–2 days)

- [ ] Tag the current state (`git tag v0-preprint-draft`). Don't submit it.
- [ ] Remove or soften claims now contradicted by the data (every row in REVIEW §T).
- [ ] Remove "we introduce the generator–verifier gap", "compute-matched" (until V4 is fixed), "cheaply decidable", "gain nowhere" and the pooled R² headline (S1).
- [ ] Decide the target paper (section 2) and the venue (section 8). Assign workstreams (section 10).

## 4. Phase 1: fix the measurement (about 1 week, one engineer)

Each item names the finding it fixes and how you know it's done.

### Generation and budgets
- [ ] **V1**: one `max_tokens` per family for **all** arms, set from a pilot at about 1.2× the 99th-percentile completion length of the longest-reasoning prompt. Log `completion_tokens >= max_tokens` per cell.
  - *Done when:* truncation is below 1% in every cell and printed in the paper.
- [ ] **V2**: the same base prompt for every arm.
  - If long CoT is a baseline, also build the pool from the CoT prompt, so self-consistency and judge run over CoT samples as in Wang et al. 2023.
  - Report "short-answer pool" and "CoT pool" as separate conditions.
- [ ] **V4**: measure the single agent at every budget a multi-agent cell uses, so no clamping remains.
  - Options: more refine rounds (up to 8), long-CoT caps up to 16×, or **budget forcing** (s1-style, `muennighoff2025simple`) to hit an exact token budget.
  - *Done when:* no comparison relies on `interp_accuracy` clamping (`metrics.py#L134`).
- [ ] **V3**: pre-specify the single-agent baseline, or choose it on calibration and score it on test. No in-sample max.
- [ ] **V5**: log wall-clock and GPU-seconds per call, and count shared prompts once for parallel samples (prefix-cached accounting). Report four cost models (angle 2).

### Judges, probe, debate
- [ ] **V6**: remove the `snippet` truncation (`arch.py` L156/L198/L317). If context forces a cut, trim the middle and keep the last 300 characters.
- [ ] **S2**: sample several correct/incorrect pairs per task (all pairs up to M = 4 per task), from more calibration tasks. Target at least 100 pairs per (model, family). Report G, AUC and d′ with CIs from a hierarchical bootstrap over tasks, then pairs.
- [ ] **S3**: if either presentation of a pair abstains, drop both. Report abstention per cell.
- [ ] **V7**: run a format-compliance check first. A judge with order consistency below 0.2 is excluded from G→η validation and reported as "cannot judge".

### Data
- [ ] **S5**: make WordCon prompts unique (more topics and lengths), and assert unique prompts and disjoint splits in `build_data.py`.
- [ ] **V8**: Logic asks "tallest/oldest"; MBPP grading uses held-out tests only, or the paper says it uses all tests.
- [ ] **R1**: set the task count in one config file (no hidden `GVGAP_LIMIT`), and write it into the results.

### Reproducibility
- [ ] **R3**: add item ID and sample index to the cache key (`engine.py#L108`), so duplicate prompts never collide.
- [ ] **R4**: set seeds per request, not per batch offset (`engine.py#L158`).
  - Write git SHA, a config hash, a data hash, the HF model revision SHA, and mlx/vllm/torch versions into every raw JSON.
  - `run.py` refuses to reuse outputs whose hashes differ (add `--force`).
  - Pin exact versions (a lockfile).
- [ ] **R5**: export every generation (prompt, text, tokens, seed, arm) to JSONL. Publish it as a GitHub release asset, Zenodo DOI or HF dataset. Run and ship `extraction_audit.py` output.
- [ ] **R6**: add `verify_refs.py` or remove the claim. Report `n_boot` honestly (10,000 everywhere, or say 2,000). Tables name their model.

### Analysis
- [ ] **S1**: evaluate each arm separately. The G→η test is the judge arm only, compared against baselines:
  - predict 0;
  - η measured at k=2, which equals G under i.i.d. sampling (N3);
  - verifier gain (`lu2025verification`);
  - the knockout bound (`chen2024provable`).
- [ ] **S4**: compute decision-rule outcomes on the **test split only**. Add leave-one-family-out and leave-one-model-out evaluation.
- [ ] **C1–C4**: rebuild the citation study (angle 7), or drop it.

### Tests
- [ ] Add one check per bug class: a truncation guard, prompt uniqueness, cache-collision regression (duplicate prompts must give distinct pool samples), and the decision rule using only test tasks.

**Phase 1 exit:** re-run the current two models on the fixed code and compare with the numbers in REVIEW V1/V2. If the headline changes, the paper changes. That is fine and expected.

## 5. Phase 2: scale the evidence (about 2–6 weeks, compute-bound)

### Models
Check exact current versions when you run.

| Role | Suggestion | Why |
|---|---|---|
| Within-family scale ladder | Qwen3 dense ~0.6B → 32B, or keep Qwen2.5 and go 0.5B → 72B | Isolate scale, as the paper intended; gives a real trend instead of 2 points |
| Second open family | Llama 3.x 8B/70B or Gemma 3 1B–27B | Reviewers will ask "is this a Qwen artifact?" |
| Reasoning model | An R1-distilled Qwen/Llama or Qwen3 thinking mode | Changes serial-vs-parallel economics (conflicting results in PRIOR_WORK §E) |
| Open-weight mid/large | e.g. gpt-oss-20b/120b | Modern agentic-scale model |
| Frontier API (1–2) | Claude / GPT / Gemini current tier | At least for the probe and judge arms; answers "does this matter for systems people deploy?" |
| Cross-model judges | Small generator plus larger or different-family judge | Angle 5; `lu2025verification` found cross-family verification better |

### Tasks
Span the verification spectrum with real verifiers and real non-verifiable tasks.

| Verification regime | Task | Verifier |
|---|---|---|
| Exact, cheap | CSP / Sudoku / ZebraLogic-style grids, with no truncation | Constraint checker |
| Execution | MBPP+ / HumanEval+ (EvalPlus), LiveCodeBench | Hidden tests |
| Formal plan validation | Blocksworld (PlanBench) | VAL validator |
| Answer-checkable math | MATH-500, AIME | Numeric match |
| Verification ≈ generation | Multi-hop QA (HotpotQA/MuSiQue) | None cheap |
| Knowledge, no verifier | GPQA-Diamond / MMLU-Pro | None cheap |

- **Scale:** at least 200 tasks per family, k ∈ {1, 2, 4, 8, 16, 32}, and 3 seeds for pools.
- **Compute:** calls ≈ tasks × families × models × about 70 calls per task (pool 32, judge sub-pools, refine, long CoT, debate, probe).
  - At roughly 0.5–1k tokens per call the full grid is on the order of 10⁸–10⁹ tokens (no measurement yet).
  - Run a 10-task pilot per family and extrapolate before buying compute.
- **Infra:**
  - vLLM with prefix caching on rented GPUs: one H100-class GPU for ≤14B, 2–4 for 70B.
  - Or hosted open-model APIs.
  - MLX on a laptop is fine for ≤7B bf16, but slow for the full grid.

## 6. Phase 3: the novel contribution (pick one core plus one or two supporting)

Each angle: the question, why it's new relative to the prior work, the design, what counts as success, and the risk.

### Angle 1 (recommended core): a signal-detection model of LLM judges
- **Question.** Does a counterbalanced pairwise probe predict best-of-k selection accuracy across k, models and tasks? Which SDT variant is right?
- **Why new.**
  - `chen2024provable` gives a bound for knockout tournaments, not a listwise prediction.
  - `cacioli2026signal` applies SDT to LLM confidence, not selection.
  - `liusie2024efficient` and `liu2025pairjudge` are methods, not forecasts.
  - DIANOIA's rules are explicitly qualitative. No one validates 2AFC→mAFC for LLM judges.
- **Design.**
  - Compare equal-variance SDT (the current Prop. 2), unequal-variance SDT (fit σ from confidence-rated z-ROC on calibration), Thurstone/Plackett–Luce listwise models, and `decarlo2012signal`'s mAFC-with-bias (position bias as a criterion shift).
  - Judge protocols: listwise, knockout tournament, pointwise scores.
  - Report k-curves (2 → 32), calibration plots, and leave-one-model/family-out error.
- **Success.** Held-out |η̂ − η| ≤ 0.05 on most cells. It must beat the baselines in S1 prospectively.
- **Risk.** Listwise judges violate independence (`zhao2025sample`). If SDT fails, measuring *how* it fails (bias, unequal variance, position effects) is still a publishable characterization.

### Angle 2: the cost model decides the verdict
- **Question.** Do "multi-agent loses at equal budget" conclusions survive when cost is completion tokens, prefix-cached tokens, FLOPs, latency or dollars?
- **Why new.** Prior equal-budget work counts tokens or thinking tokens (`tran2026single`, `kim2025science`, `wang-etal-2024-reasoning-token`, `mirzaei2026sample`). With prefix caching and batching, parallel sampling can be cheap and fast.
- **Design.** Measure every run under all cost models on the same hardware and serving stack, and map where the sign of the lift flips.
- **Success.** A clear "which cost model you use changes the answer" figure plus a recommendation.
- **Risk.** Low. This is solid measurement work and a good supporting section.

### Angle 3: head-to-head benchmark of pre-deployment diagnostics (pre-registered)
- **Question.** Which cheap diagnostic best predicts whether an ensemble beats a same-budget single agent on new tasks and models?
- **Candidates.** Your G/η, verifier gain (`lu2025verification`), plurality margin d_V (`chen2024more`), oracle gap + signal fidelity (`hu2026oracle`), DIANOIA rules (`yang2026dianoia`), a Kim-style regression (`kim2025science`), and vote entropy (`hu2026statistical`, which found it fails).
- **Why new.** These papers each propose a diagnostic. None compares them prospectively on shared held-out cells; this audit found no such comparison, which is a negative search result.
- **Design.** Freeze the models, tasks and predictions, publish the hash of the predictions file (OSF or a GitHub release) **before** running the held-out cells, then score.
- **Success.** A ranking with CIs. Even "nothing beats headroom × verifier-type" is a useful result.
- **Risk.** Moderate. Implementing others' diagnostics carefully takes time.

### Angle 4: measure verification asymmetry directly
- **Question.** Can you put a number on Brown's generator–verifier gap for a (model, task) pair? For example: tokens or compute the model needs to verify a candidate at a given accuracy, versus to generate a correct one.
- **Why new.** Brown and Wei describe it qualitatively; DIANOIA's R1–R4 are qualitative. A measured "asymmetry index" that predicts η and matched lift would be a contribution.
- **Design.** Per task, measure solve cost at a target accuracy vs. verify cost (pairwise and pointwise) at the same accuracy. Regress η and lift on the index.
- **Risk.** Defining "equal accuracy" cleanly. Pilot on CSP, code and QA first.

### Angle 5: cross-model verification economics
- **Question.** When does "cheap generator + stronger or different-family judge" beat "one strong agent" at equal dollars? Does cross-model G predict it?
- **Why new.** `lu2025verification` shows cross-family verification helps; `zhang2025stop` argues for heterogeneity. Nobody gives a cost-matched forecasting rule.
- **Design.** Generator × judge matrix across sizes and families; price everything; predict with cross-model G.
- **Success.** A deployable rule ("use a judge from family X when G_cross > t").

### Angle 6: when does serial beat parallel? Reconcile the conflicting literature
- **Question.** `mirzaei2026sample` finds refinement loses at 1.5–7B; `sharma2025sequential` finds sequential wins for stronger models; `wunderlich2026multi` finds debate and MoA win at 70B; `yang2025revisiting` finds debate helps small models on hard math. Where is the crossover?
- **Design.** Use your grid (angle 1 models × tasks) and plot the lift sign against model capability and task difficulty.
- **Success.** A capability/difficulty phase diagram that explains the conflicts. Reviewers love a reconciliation.

### Angle 7: citation verification, done right (deployed case study)
- **Question.** How often do current models fabricate references? How well do model critics, registries and hybrids catch them, per dollar?
- **Fixes.**
  - Truth by DOI + normalized title + first-author surname + year (±1), from CrossRef/OpenAlex.
  - Two annotators manually audit every positive and a sample of negatives; report κ.
  - A selector that does **not** use the ground-truth lookup.
  - At least 200 topics, frontier models, and a retrieval-augmented condition.
  - Counterbalanced, well-powered G.
- **Before writing.** Search the reference-hallucination literature. This audit did not survey it.

### Recommended package
- **TMLR / journal:** Angle 1 as the core, Angle 3 as the evaluation protocol (pre-registered), Angle 2 as a section, and Angle 6 as the discussion that reconciles the literature.
- **Workshop now:** Angle 1 on the fixed two-model grid plus two or three more model sizes.

## 7. Phase 4: rewrite and reposition (about 1–2 weeks)

- [ ] **Title and abstract** only claim what the new results show. No pooled R², no "cheaply decidable" and no "gain nowhere", unless the new data earns them.
- [ ] **Related work** gets 4 paragraphs:
  1. generator/verifier gaps (N1);
  2. coverage/selection decompositions (N2: DIANOIA/PRISM, Hu, Weaver, Saunders);
  3. equal-budget multi-agent studies (N6: Mirzaei, Tran & Kiela, Kim, Hu et al., Wang EMNLP'24, Choi, Singhi);
  4. pairwise comparison and SDT (N3/N4).

  End each with one sentence saying precisely what you add. Use the shortlist in [PRIOR_WORK.md](PRIOR_WORK.md).
- [ ] **Framework.**
  - Prop. 1 becomes a definition, citing PGR/DIANOIA.
  - Prop. 2 is "the classical equal-variance SDT 2AFC→mAFC relation (Thurstone 1927; Green & Swets 1966; Hacker & Ratcliff 1979)", with the closed form instead of Monte Carlo.
  - Add the G = η(2) identity (N3).
- [ ] **Results.**
  - Per-arm results.
  - Truncation rates.
  - CIs on every diagnostic.
  - Leave-one-out prediction.
  - Cost-model sensitivity.
  - The conflicting-results reconciliation.
- [ ] **Limitations.** Keep the candor. That part is already good.
- [ ] **Reproducibility statement.** Commit SHA, data DOI, generations DOI, one command per figure.
- [ ] **AI disclosure** in the acknowledgments: name the system and the sections it produced (IEEE requires this; most ML venues ask for it too).
- [ ] **Contribution statement (CRediT)**: who did conceptualization, methodology, software, analysis and writing, stated accurately. From now on, commit under each person's own name so the git history is a truthful, contemporaneous record of who did what. The current history has one committer plus AI co-author trailers.
- [ ] **Authors:** ORCIDs and short bios (IEEE Access).

## 8. Phase 5: venue plan

| Venue | Fit | Requirements and notes |
|---|---|---|
| **TMLR** | Best fit after the fixes | Rolling submission; claims-supported criterion; novelty not required |
| Workshop at NeurIPS/ICLR/ICML (reasoning, test-time compute, agents) | Fast feedback on Angle 1 | Short papers; often non-archival, so you can still go to TMLR later |
| ACL Rolling Review → ACL/EMNLP (main or Findings) | If the empirical study gets broad (angles 3 + 6) | Needs a strong NLP-audience framing |
| NeurIPS/ICLR main track | Only with a strong new result: SDT beats all diagnostics prospectively, or a crisp serial-vs-parallel law | High novelty bar |
| IEEE Access | Fallback | 20% acceptance; binary accept/reject in ~4–6 weeks; APC US$2,160; the **IEEE Access template** is mandatory; author bios; AI disclosure; reviewers check contribution, soundness, sufficiency of references, and conclusions supported by data |

Post to arXiv first. If targeting IEEE, add the required "submitted to the IEEE" notice.

**Archival vs. non-archival.** Many workshops are non-archival: there are no proceedings. Use them for feedback and visibility. If you need the work to count as a *published* peer-reviewed article, aim for an archival venue: TMLR, ACL/EMNLP (incl. Findings), a main conference, or IEEE Access.

## 8b. Make the work usable by others (this is what creates impact)

- Package the diagnostic as a small tool: "probe your judge on 30–50 labelled tasks → predicted η and a build/don't-build call with CIs". Give it a README, a pip install and one worked example.
- Release the benchmark and all generations with a DOI (Zenodo or HF), so others can compare against your numbers.
- Write a short technical blog post or thread with the one figure that matters. Give talks: reading groups, meetups, workshop posters.
- Serve as a reviewer: TMLR, ACL Rolling Review and workshop program committees take volunteers. Keep the confirmation emails.
- Track who uses or cites the tool (GitHub dependents, forks, citations). Talk to those people; their feedback is the next paper.
- Integrity: no citation trading or self-citation padding, no predatory venues, and every contribution statement exactly true.

## 9. Pre-mortem: what reviewers will say, and your answer

| Objection | Answer after this roadmap |
|---|---|
| "This is DIANOIA/PRISM and Hu 2026." | Theirs are descriptive or qualitative. Ours is a prospective, quantitative SDT forecast, compared head-to-head (angle 3) |
| "The gains/losses are truncation and prompt artifacts." | One cap and one prompt for all arms; truncation below 1% and reported |
| "Two tiny models." | 6+ models from 3+ families, up to 70B plus a frontier API model |
| "G is noise." | 100+ pairs per cell, hierarchical CIs, abstention-safe counterbalancing |
| "R² is inflated by mixing arms." | Per-arm results, baselines, leave-one-out, pre-registered |
| "Tokens aren't compute." | Four cost models (angle 2) |
| "Prop. 2 is Thurstone." | Cited as such. The contribution is empirical validation and the unequal-variance/bias extension for LLM judges |
| "Not really held out." | Predictions hashed and published before the held-out runs |
| "Synthetic toy tasks." | Real verifiers (tests, VAL) plus tasks with no cheap verifier |
| "Was this AI-generated?" | Disclosed in the acknowledgments; every number regenerates from released generations |

## 10. Team workstreams

| Workstream | Owns | First deliverable |
|---|---|---|
| WS1 Infra and correctness | Phase 1 code fixes, provenance, tests | Fixed code; two-model re-run; diff against old numbers |
| WS2 Experiments and compute | Phase 2 grid, serving stack, releasing generations | Pilot cost estimate, then the full grid |
| WS3 Theory and analysis | Angle 1 SDT variants, baselines, pre-registration, leave-one-out | Pre-registration document plus the analysis notebook |
| WS4 Literature and writing | PRIOR_WORK.md, related work, rewrite, venue compliance | New related-work section and revised abstract |

### Master checklist (copy into issues)
- [ ] Phase 0 triage (§3)
- [ ] V1 · V2 · V3 · V4 · V5 · V6 · V7 · V8
- [ ] S1 · S2 · S3 · S4 · S5 · S6
- [ ] R1 · R2 · R3 · R4 · R5 · R6
- [ ] C1–C4 (rebuild or drop)
- [ ] N1–N8 addressed in related work and framework
- [ ] X1–X3 claims rescoped
- [ ] T: every contradicted sentence fixed
- [ ] P: template, bios, ORCID, AI disclosure
- [ ] Phase 2 grid (models × tasks × k)
- [ ] Core angle plus supporting angles (§6)
- [ ] Rewrite (§7) and venue (§8)
