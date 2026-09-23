# Prior work the paper must engage with

This list was found during the audit, and every entry was checked on arXiv, the proceedings page or CrossRef on 2026-09-23. BibTeX keys refer to [`prior_work.bib`](prior_work.bib), which lists full author lists (the current `refs.bib` truncates four of them).

**Overlap** says how much a work pre-empts this paper:
- **identical**: same quantity or result.
- **strong**: a reviewer will say "this was done".
- **partial**: same territory, different angle.
- **context**: cite for positioning.

**Action** says what to do with it:
- **Differentiate**: explain precisely how your contribution differs.
- **Cite**: one-line citation in related work.
- **Compare**: run it as a baseline or head-to-head.

## Must-cite shortlist (25)

If you only read one section, read this one. Items marked ★ are the ones reviewers will name first.

| # | Work | Key | Why it matters | Action |
|---|---|---|---|---|
| ★1 | Brown 2024, talk "The Generator-Verifier Gap" (Simons Institute) | `brown2024gvgap` | The name and the thesis | Cite; do not claim to "introduce" the gap |
| ★2 | Song et al., ICLR 2025, *Mind the Gap* | `song2024mind` | Formal "generation-verification gap" for LLMs | Differentiate your G |
| ★3 | Yang et al., EMNLP 2026, *DIANOIA* (v1 *PRISM*) | `yang2026dianoia` | E[Q] = C_K·η, "exploration headroom C_K − p", verifier-type rules, GSM8K/MBPP, budget-matched | Differentiate + compare |
| ★4 | Hu 2026, *Oracle Gap and Signal Fidelity* | `hu2026oracle` | Selector gain bounded by oracle gap, then signal fidelity; "pre-deployment diagnostic" | Differentiate + compare |
| ★5 | Mirzaei 2026, *Sample More, Reflect Less* | `mirzaei2026sample` | 1.5B–7B; debate / self-picked best-of-N / Self-Refine vs. sampling at equal tokens; bootstrap + correction | Differentiate (closest empirical precedent) |
| ★6 | Tran & Kiela 2026 | `tran2026single` | Single agent ≥ multi-agent at equal thinking-token budgets | Cite; kills the "not standard practice" framing |
| ★7 | Kim et al. 2025/26, *Towards a Science of Scaling Agent Systems* | `kim2025science` | Predictive model of multi-agent vs. single agent under budgets (v1 CV R² 0.513; v3 0.373–0.413) | Compare |
| ★8 | Saad-Falcon et al., NeurIPS 2025, *Weaver* | `saadfalcon2025shrinking` | "Shrinking the generation-verification gap"; gap = Pass@K − selected accuracy | Cite |
| ★9 | Thurstone 1927; Mosteller 1951 | `thurstone1927law`, `mosteller1951remarks` | Your Proposition 2 is Thurstone Case V / probit paired comparison | Cite at Prop. 2 |
| ★10 | Green & Swets 1966; Hacker & Ratcliff 1979; Macmillan & Creelman 2005; DeCarlo 2012 | `green1966signal`, `hacker1979revised`, `macmillan2005detection`, `decarlo2012signal` | 2AFC → mAFC under equal-variance signal detection theory, which is your pairwise → listwise step | Cite at Prop. 2 |
| 11 | Chen et al., NeurIPS 2025, *Provable Scaling Laws for Test-Time Compute* | `chen2024provable` | p_comp = your q; knockout-tournament bound | Compare (pairwise → best-of-N) |
| 12 | Lu et al. 2025, *When Does Verification Pay Off?* | `lu2025verification` | "Verifier gain" predicts rejection-sampling gains; 37 models including Qwen2.5-0.5B/1.5B | Compare |
| 13 | Singhi et al., COLM 2025, *When To Solve, When To Verify* | `singhi2025solve` | At matched compute, self-consistency beats generative verifiers | Cite |
| 14 | Choi et al., NeurIPS 2025, *Debate or Vote* | `choi2025debate` | Voting explains most debate gains | Cite |
| 15 | Huang et al., ICLR 2024 (already in refs) | `huang2023cannot` | Table 7: debate < self-consistency at equal responses | Cite for this result too |
| 16 | L. Chen et al., NeurIPS 2024, *Are More LLM Calls All You Need?* | `chen2024more` | d_V: voting helps iff the truth is the plurality, i.e. your "plurality alignment" | Cite |
| 17 | Saunders et al. 2022, *Self-critiquing models* | `saunders2022self` | GD gap vs. oracle best-of-2; "compare to P ⊆ NP" | Cite |
| 18 | Stechly, Valmeekam, Kambhampati 2024 | `stechly2024self` | Self-verification collapses; sound external verifiers help | Cite |
| 19 | Huang et al., ICML 2025, *Is Best-of-N the Best of Them?* | `huang2025best` | Coverage and imperfect reward models in best-of-N | Cite |
| 20 | Zhang et al., ICLR 2025, *Generative Verifiers* | `zhang2024generative` | Verifier quality in best-of-N | Cite |
| 21 | Liu et al. 2025, *PairJudge RM* | `liu2025pairjudge` | Pairwise judge plus knockout for best-of-N | Cite / compare |
| 22 | Cacioli 2026, *LLMs as Signal Detectors* | `cacioli2026signal` | SDT applied to LLMs; **unequal variance** (z-ROC slopes 0.52–0.84) | Cite; test unequal-variance SDT |
| 23 | Wang et al., EMNLP 2024, *Reasoning in Token Economies* | `wang-etal-2024-reasoning-token` | Budget-aware evaluation; complex strategies' gains are mostly compute | Cite |
| 24 | Smit et al., ICML 2024; Wang et al., ACL 2024 | `smit2023should`, `wang2024rethinking` | Debate vs. strong single-agent baselines | Cite |
| 25 | Wei 2025, "Asymmetry of verification and verifier's law" | `wei2025asymmetry` | Thesis restated | Cite |

## By theme (full list)

### A. Generator/verifier gaps and self-verification

| Work | Key | Overlap | Note |
|---|---|---|---|
| Brown 2024 talk | `brown2024gvgap` | identical (name, thesis) | "When verifying a solution is easier than generating it … scaling up inference time compute gets better results" |
| Song et al., ICLR 2025 | `song2024mind` | strong | Generation-verification gap; small models (0.5B) have gap ≤ 0 |
| Saunders et al. 2022 | `saunders2022self` | strong | GDC gaps; compute-equalized baselines |
| Weaver, NeurIPS 2025 | `saadfalcon2025shrinking` | strong | Weak-verifier ensembles close the gap |
| Jiang et al., AAAI 2025, *SELF-[IN]CORRECT* | `jiang2024self` | strong | Discrimination among own samples ≤ generation |
| West et al., ICLR 2024, *Generative AI Paradox* | `west2023generative` | partial | Models generate better than they discriminate |
| Li et al., ICLR 2024, generator-validator consistency | `li2023benchmarking` | partial | |
| Sun et al. 2025, solver-verifier gap | `sun2025theoretical` | partial | |
| Feng et al. 2026, *Peer-Predictive Self-Training* | `feng2026peer` | context | Already uses the term "GV-Gap" |
| Davidson et al. 2026, *factual generation-verification gap* | `davidson2026future` | context | |
| Swamy et al., ICLR 2026, *All Roads Lead to Likelihood* | `swamy2025roads` | context | Attributes RL gains to generation-verification gaps |
| Huang et al., ICLR 2025, *Sharpening* | `huang2024self` | context | |
| Stechly et al. 2024 | `stechly2024self` | strong | |
| Kambhampati et al., ICML 2024, *LLM-Modulo* | `kambhampati2024plan` | partial | External verifiers in the loop |
| Zhang et al., Findings ACL 2024, *Small LMs Need Strong Verifiers* | `zhang2024small` | strong | Directly relevant to 0.5–1.5B |
| Kamoi et al., TACL 2024 | `kamoi2024actually` | strong | Self-correction needs reliable external feedback |
| Irving et al. 2018, *AI safety via debate* | `irving2018safety` | context | |
| Kirchner et al. 2024, *Prover-Verifier Games* | `kirchner2024prover` | context | |
| Sunkaraneni et al. 2026, *Agentic Systems as Boosting Weak Reasoning Models* | `sunkaraneni2026agentic` | strong | Separates coverage from selection; needs execution/tests/proof checks; comparator must beat 1/2 |

### B. Coverage, selection and best-of-N

| Work | Key | Overlap | Note |
|---|---|---|---|
| DIANOIA / PRISM, EMNLP 2026 | `yang2026dianoia` | strong | Coverage × efficiency identity; rules R1–R4 are explicitly "qualitative task-structure hypotheses", which is room for a quantitative version |
| Hu 2026, *Oracle Gap and Signal Fidelity* | `hu2026oracle` | strong | LiveCodeBench, MATH-L5, GPQA-Diamond |
| Bay & Yearick 2026, *When More Sampling Hurts* | `bay2026more` | partial | Modal and correlation ceilings of test-time scaling |
| J. Chen 2026, *Co-Failure Ceiling* | `chen2026combining` | partial | Selection capped by joint failure rate; 67 frontier models |
| Huang et al., ICML 2025, *Is Best-of-N the Best of Them?* | `huang2025best` | partial | |
| Chen et al., NeurIPS 2025, *Provable Scaling Laws* | `chen2024provable` | identical (q = p_comp) | |
| PairJudge RM 2025 | `liu2025pairjudge` | partial | |
| Generative Verifiers, ICLR 2025 | `zhang2024generative` | partial | |
| Frick et al. 2024, *How to Evaluate Reward Models (PPE)* | `frick2024evaluate` | partial | Best-of-K curves vs. oracle |
| Zhao et al., ICML 2025, *Sample, Scrutinize and Scale* | `zhao2025sample` | partial | Comparing across responses helps, which violates your independence assumption |
| Zhou et al., ICLR 2026, *Variation in Verification* | `zhou2025variation` | partial | |
| Mukherjee et al. 2025, *Verification via Optimal Transport* | `mukherjee2025test` | partial | Coverage + ROC |
| Liu 2026, *LLMs as a Jury* | `liu2026jury` | partial | Fraction of oracle gap captured; self-scoring ≈ 0% |
| Lu et al. 2025, *When Does Verification Pay Off?* | `lu2025verification` | strong | |
| Singhi et al., COLM 2025 | `singhi2025solve` | strong | |
| Burns et al., ICML 2024, *Weak-to-Strong* (PGR) | `burns2023weak` | partial | η has PGR's form; already in `refs.bib`, never cited |
| Schaeffer et al. 2025, *Monkeys power laws* | `schaeffer2025monkeys` | context | Forecasting pass@k cheaply |

### C. Pairwise comparison, SDT and statistics

| Work | Key | Overlap | Note |
|---|---|---|---|
| Thurstone 1927 | `thurstone1927law` | identical (Prop. 2) | Case V |
| Mosteller 1951 | `mosteller1951remarks` | identical | Least-squares paired comparisons |
| Green & Swets 1966 | `green1966signal` | identical | 2AFC |
| Hacker & Ratcliff 1979 | `hacker1979revised` | identical | mAFC d′ tables |
| Macmillan & Creelman 2005 | `macmillan2005detection` | identical | Textbook |
| DeCarlo 2012 | `decarlo2012signal` | identical | mAFC with bias; ML/Bayesian estimation. Use it to fit bias too |
| Hanley & McNeil 1982 | `hanley1982meaning` | identical (q = AUC) | |
| List & Goodin 2001 | `list2001epistemic` | identical (plurality alignment) | Plurality Condorcet |
| Liusie et al., EMNLP 2024, Product of Experts | `liusie2024efficient` | partial | Gaussian model of LLM pairwise judgments |
| Cacioli 2026 | `cacioli2026signal` | partial | Unequal variance in LLMs |
| Wu et al., ICLR 2025, *Inference Scaling Laws* | `wu2024inference` | identical (voting limit) | Thm 1: majority vote converges to "truth is the mode" |
| L. Chen et al., NeurIPS 2024 | `chen2024more` | strong | |

### D. Multi-agent vs. single agent at equal budget

| Work | Key | Overlap | Note |
|---|---|---|---|
| Mirzaei 2026 | `mirzaei2026sample` | strong | Closest empirical precedent |
| Tran & Kiela 2026 | `tran2026single` | strong | Qwen3-30B-A3B, R1-Distill-70B, Gemini 2.5 |
| Kim et al. 2025/26 | `kim2025science` | strong | Quantitative predictor; saturation near 45% single-agent accuracy |
| Hu, Shen & Lakshmipathi 2026 | `hu2026statistical` | strong | Pre-debate signals fail to predict when debate helps: "safe, not useful" |
| Wang et al., EMNLP 2024, *Token Economies* | `wang-etal-2024-reasoning-token` | strong | |
| Smit et al., ICML 2024 | `smit2023should` | strong | |
| Zhang et al. 2025, *Stop Overvaluing MAD* | `zhang2025stop` | strong | Also argues for model heterogeneity |
| Yang et al. 2025, *MAD as Test-Time Scaling* | `yang2025revisiting` | strong | **Conflicts**: debate helps smaller models on hard math |
| Wunderlich et al., ACL 2026 SRW | `wunderlich2026multi` | strong | **Conflicts**: debate and MoA Pareto-better than self-consistency at 70B |
| Choi et al., NeurIPS 2025 | `choi2025debate` | identical (debate ≈ vote) | |
| Jwalapuram et al. 2026, *Illusion of Multi-Agent Advantage* | `jwalapuram2026illusion` | partial | |
| Bertalanič & Fortuna 2026, *Cost of Consensus* | `bertalanic2026cost` | partial | 7–8B: debate costs 2.1–3.4× the tokens for ≤ accuracy |
| Xu et al. 2026, strong single-agent baseline | `xu2026rethinking` | partial | |
| Gao et al. 2025, *SAS or MAS? Why Not Both* | `gao2025single` | partial | Multi-agent advantage shrinks with capability |
| Wang et al., ACL 2024, *Rethinking the Bounds* | `wang2024rethinking` | partial | |
| Li et al. 2025, *Rethinking Mixture-of-Agents* | `li2025rethinking` | context | |
| Zhao et al. 2026, *When Does MA Collaboration Help? (entropy)* | `zhao2026multi` | partial | |
| Liu et al. 2026, *Phase Transition for Budgeted MA Synergy* | `liu2026phase` | partial | |
| Hong 2026, difficulty-aware topology selection | `hong2026learning` | partial | Per-problem predictor under matched budgets |
| Pappu et al., ICML 2026, *Teams Hold Experts Back* | `pappu2026multi` | context | |
| Kaesberg et al., Findings ACL 2025, *Voting or Consensus?* | `kaesberg2025voting` | partial | |
| Wu et al. 2025, *Can LLM Agents Really Debate?* | `wu2025agents` | partial | |
| Wynn et al. 2025, *Talk Isn't Always Cheap* | `wynn2025talk` | partial | |
| Qian 2026, *What Does MAD Actually Change?* | `qian2026multi` | context | |
| Motger et al. 2026, MAD survey | `motger2026multi` | context | 141 studies: designs chosen "by convention", which partly supports your motivation |
| Anthropic 2025, multi-agent research system | `anthropic2025multiagent` | context | Token usage explains most of the variance |

### E. Serial vs. parallel test-time compute

| Work | Key | Overlap | Note |
|---|---|---|---|
| Snell et al. (already in refs) | — | strong | Optimal serial/parallel split depends on difficulty |
| Sharma & Chopra 2025, *The Sequential Edge* | `sharma2025sequential` | partial | **Conflicts**: sequential wins at matched compute for stronger models |
| Zeng et al. 2025, *Revisiting o1-like TTS* | `zeng2025revisiting` | partial | |
| Muennighoff et al. 2025, *s1* | `muennighoff2025simple` | context | Budget forcing |
| Liu et al. 2025, *Can 1B Surpass 405B?* | `liu2025surpass` | context | |

## Conflicting results you must reconcile

The paper's claims have to be scoped against these:
- Wunderlich et al. 2026: at equal compute on 70B, debate beats self-consistency by +1.3 points and mixture-of-agents by +2.7.
- Yang et al. 2025: debate helps relatively more on hard math and for smaller models ("doubles AIME accuracy for Qwen2.5-3B").
- Sharma & Chopra 2025: sequential refinement beats parallel self-consistency at matched compute in 95.6% of configurations.

A paper that explains *when* each of these holds is more publishable than one that says "multi-agent loses".
