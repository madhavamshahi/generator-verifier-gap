# External audit: start here

An audit of `main` at `99ff91a` ("The Generator–Verifier Gap Predicts When Multi-Agent LLM Architectures Beat a Compute-Matched Single Agent"), done 2026-09-23. It merges two independent reviews. Every numeric claim is re-checked against the shipped data, and the prior work was checked on arXiv, proceedings pages and CrossRef.

## Verdict

**Don't submit this version. There is a real paper inside it, but a narrower one.**

- **What holds up.** The code is clean and every table regenerates from `results/raw/`. The question — should you build the multi-agent system, decided from a cheap test beforehand — is worth asking.
- **The headline results rest on confounds.**
  - Ensemble members were cut off at 320 tokens while the baseline got 2–4× the budget and a stronger prompt. That explains most of the "gains flip sign" result.
  - R² = 0.84 barely uses G: ignore G entirely and R² is still 0.82.
  - Every value of G is statistically indistinguishable from zero.
  - The citation study's η = 1 is true by definition.
- **The novelty claims are contradicted by 2024–2026 work, none of it cited.**
  - Brown 2024 and Song 2025 on the generator–verifier gap.
  - DIANOIA/PRISM and Hu 2026 on coverage × selection.
  - Mirzaei, Tran & Kiela and Kim et al. on equal-budget multi-agent evaluation.
  - Proposition 2 is classical Thurstone / signal-detection theory.
- **Plagiarism.** No copied text found. Several ideas are uncredited.
- **Readiness: about 20% of the way to a paper that survives competent review** (judgment estimate; breakdown in [ROADMAP.md §1](ROADMAP.md#1-where-you-are-about-20-of-the-way)).

## What's in this folder

| File | What it is |
|---|---|
| [REVIEW.md](REVIEW.md) | Every problem, with an ID (V/S/R/C/N/X/T/P), severity, evidence and code line links |
| [ROADMAP.md](ROADMAP.md) | Readiness scorecard, which paper to write, phase-by-phase fixes, 7 research angles for novelty, models and tasks to add, venue plan, reviewer pre-mortem, team workstreams |
| [PRIOR_WORK.md](PRIOR_WORK.md) | ~80 related works grouped by theme, with overlap ratings and a must-cite shortlist |
| [prior_work.bib](prior_work.bib) | BibTeX for all of them. arXiv entries come from the arXiv API with full author lists; classic papers were checked on CrossRef |
| [audit.py](audit.py) | Reproduces every number in REVIEW.md: `python3 review/audit.py` (no GPU, about 10 s) |
| [O1A_PLAN.md](O1A_PLAN.md) | How the research maps onto the O-1A criteria (checked against the USCIS Policy Manual), corrections to the O-1A report the team shared, and integrity guardrails. Not legal advice |

## The ten things to fix first

1. **V1** One token cap for every arm, and fewer than 1% of outputs truncated.
2. **V2** One base prompt for every arm; ensembles built over CoT samples too.
3. **V6** Judges, probe and debate see full candidates, not the first 700 or 500 characters.
4. **S2 / S3** 100+ probe pairs per cell, pairs dropped when either order abstains, and CIs reported.
5. **S1** Validate G→η on the judge arm only, against baselines. Drop the pooled R² headline.
6. **V4** Measure the single agent at every budget (no clamping), or stop saying "compute-matched".
7. **R3 / R4 / R5** Fix the cache-key collision, add provenance hashes and pinned versions, and ship all generations.
8. **N1–N8** Rewrite related work and positioning. Don't claim to introduce the gap; cite Thurstone/SDT at Prop. 2.
9. **X1** Add 7B–70B models from two or more families plus one frontier API model.
10. **P** IEEE Access template, author bios and AI disclosure, if you still target IEEE. TMLR is the better target (ROADMAP §2).

## How this audit was done

- Read every source file, the LaTeX, the bib, the README, the submission package and all results files.
- Re-ran the analysis chain and the estimator tests. Both reproduce.
- Replicated the probe's pair sampling to measure what the judge could see.
- Checked every related paper against its arXiv, proceedings or CrossRef record. Checked the IEEE Access and TMLR policies on their live pages.
- **Not done:** re-running generation. The generation cache isn't shipped, and a fresh run needs model weights. So V6 (truncated judge inputs) is estimated from token counts, not measured on the texts.
