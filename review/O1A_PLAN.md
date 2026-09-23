# O-1A alignment plan for the generator–verifier project

**Visibility:** this repo is public. USCIS officers and the independent experts who write your letters can read this file. The team chose to keep it with the review; move it somewhere private if that changes.

This is research planning, not legal advice. Have an immigration attorney confirm every legal point before you rely on it.

## 0. A note on visibility

Adjudicators and experts may look at the paper's repo. A public plan that maps the paper to visa criteria, next to a public critique of the paper, invites the Final Merits question: did the field recognize this work, or was the evidence manufactured? The strongest answer is work others actually use.

## 1. What a research program can and can't cover

The eight O-1A criteria below are from the USCIS Policy Manual, Vol. 2, Part M, Ch. 4, checked 2026-09-23. You need at least 3, then a "totality" review.

| # | Criterion (Policy Manual wording, shortened) | Can this research legitimately support it? | How |
|---|---|---|---|
| 1 | Nationally or internationally recognized prizes or awards | Rarely | A best-paper award. You can't plan for one |
| 2 | Membership in associations requiring outstanding achievements | No | Not produced by a paper |
| 3 | Published material *about you* in professional, trade or major media | Maybe | Only if independent outlets cover your work: newsletters, trade press. Your own blog doesn't count |
| 4 | Judge of the work of others | **Yes** | Completed peer reviews: TMLR, ACL Rolling Review, workshop committees, IEEE Access. Keep the confirmation emails |
| 5 | Original contributions of **major significance** | **Only with real adoption** | Independent use of your tool or benchmark, citations, field-normalized citation percentiles, independent expert letters |
| 6 | Authorship of scholarly articles in professional journals or major media | **Yes, once published** | The Manual counts peer-reviewed proceedings at recognized conferences. Prefer archival venues (§3) |
| 7 | Critical or essential role for a distinguished organization | Not from the paper | Comes from your job or startup |
| 8 | High salary | No | Comes from your job |

**Bottom line.** One paper can't fulfil "every criterion". A research program can realistically support #4 and #6, and #5 if the work is actually adopted, with #3 as a bonus. Final Merits looks at caliber and impact, not just the count.

## 2. What this paper is worth for O-1A today

- **As-is it's a liability, not an asset.**
  - IEEE Access accepts about 20% of submissions. Its reviewers check contribution, soundness, whether the references are sufficient, and whether the conclusions follow from the data. This version fails on several of those, and it would be returned before review for the wrong template, no author bios and no AI disclosure.
  - Even if it were accepted, weak work in a fast journal adds little at Final Merits. A public critique or a correction would subtract.
- **The AI-authorship question matters twice.**
  - The venue requires disclosure (IEEE's policy is explicit).
  - The petition's evidence is supposed to show *your* extraordinary ability. Using AI tools as an assistant is fine if disclosed. Claiming sole credit for work an AI agent largely produced is not.
- **Co-author declarations and CRediT statements must be literally true.** A willful misrepresentation of a material fact in an immigration filing can lead to denial and a permanent inadmissibility finding (INA §212(a)(6)(C)(i)). Ask your attorney. This is the single biggest risk to manage.

## 3. The plan: roadmap phases mapped to evidence (rough, assumes things go well)

| When | Research work ([ROADMAP.md](ROADMAP.md)) | Evidence it produces |
|---|---|---|
| Month 0–1 | Phase 0–1: fix the confounds and bugs, re-run | Nothing to file yet. This protects you later |
| Month 1–3 | Phases 2–3: models and tasks grid; core angle (SDT judge predictor, pre-registered) | Your own contribution, documented in commits under your name |
| Month 3 | arXiv preprint, code tagged v1.0, generations and benchmark with a DOI, the pip tool | Priority date; something others can use (#5) |
| Month 3–4 | Submit to an **archival** peer-reviewed venue (§4) | Toward #6 once accepted |
| Month 3–8 | Review for TMLR, ACL Rolling Review, workshops; talks, poster | #4 (completed reviews), visibility |
| Month 4–9 | Track real adoption (dependents, forks, citations, issues from other groups); help adopters | #5, if it happens |
| Month 8–10 | Independent experts who used or cited the work write in their own words | Final Merits support |
| Month 10+ | Attorney assembles and files | — |

The bottleneck is not the publication date. It is whether independent people **use** the work, and that comes from the work being correct and useful.

## 4. Venue choice with an O-1A lens

| Venue | Archival? | Notes |
|---|---|---|
| TMLR | Yes (journal on OpenReview) | Accepts papers on evidence, not novelty; strong ML reputation; public reviews |
| ACL/EMNLP main or Findings (via ACL Rolling Review) | Yes | Peer-reviewed proceedings at recognized conferences, which the Policy Manual counts |
| NeurIPS/ICLR/ICML main track | Yes | Strongest signal. Needs a strong new result |
| IEEE Access | Yes | Indexed (SCIE, Scopus), 20% acceptance, 4–6 weeks, US$2,160 APC, strict template. Fine *if the paper is sound* |
| NeurIPS/ICLR workshops | Often **no** | Great for feedback and visibility. Non-archival workshop papers may not count as published articles; confirm with your attorney |
| arXiv only | Not peer-reviewed | Establishes priority and gathers citations. Does not replace a peer-reviewed venue |

## 5. Documenting individual contributions in a three-person team

- Split ownership now: for example, one person owns infrastructure and correctness, one owns the SDT theory and analysis, one owns experiments and the tool (ROADMAP §10). Each person commits under their own name.
- Keep dated design notes and experiment logs per person.
- Write a CRediT statement in the paper that matches the git history.
- Co-author declarations later must describe exactly what the history shows. Don't use templated "solely responsible" language unless it's true.

## 6. Corrections to the O-1A report you pasted

- **PA-2025-02** (8 Jan 2025) is real. It clarifies O-1A evidence, adds examples for AI and other critical and emerging technologies, and says a company you own may file the petition. It does **not** name LLMs or multi-agent systems specifically.
- **"A preprint with 50+ citations is exceptional evidence"** is a rule of thumb, not policy.
- **"IEEE Access is universally accepted by USCIS"** is overstated. It is a legitimate indexed journal. What matters at Final Merits is the paper's caliber and impact.
- **The case vignettes** (600 citations as 12th author; 10k GitHub stars approved without an RFE) are anecdotes, not adjudication data.
- **The 11-month timeline** assumes acceptance and adoption happen on schedule. Plan for slippage.
- **The example co-author declaration and expert-letter wording** ("solely responsible", "reduced hallucination by 40%") are illustrative. Real letters must state only facts that are true and measured.

## 7. Guardrails (they protect the petition)

1. Disclose AI assistance wherever the venue requires it. Never misstate who did the work.
2. No citation trading, citation rings, self-citation padding or paper mills. Adjudicators and experts can spot them.
3. Every number in the paper regenerates from released data. The audit shows the pipeline can already do this.
4. Expert letters come from genuinely independent experts, in their own words.
5. Strategy documents like this one are best kept private.
