# Cover letter — IEEE Access

Dear Editor,

We submit *The Generator--Verifier Gap Predicts When Multi-Agent LLM
Architectures Beat a Compute-Matched Single Agent* for consideration in IEEE
Access.

Multi-agent LLM systems are deployed widely and evaluated inconsistently:
published gains are usually measured against a single model call rather than
against one agent given the same token budget. We show that this one
substitution removes 47% of reported gains in our own suite, and that 56% of
all configurations we test are net negative once compute is matched.

The paper's contribution is not the negative result but a prospective test.
A k-agent ensemble's accuracy decomposes exactly into coverage and selection
efficiency, and we give a latent-score model that turns a cheap pairwise
probe --- the generator--verifier gap --- into a parameter-free prediction of
selection efficiency. Across 540 tasks, 6 families, 2 model scales and 8
architectures with exact token accounting over 21,847 generations,
diagnostics measured on 30 calibration tasks predict held-out lift at
R^2 = 0.84 with no fitted parameters and without the multi-agent system ever
being built. A deployed citation-generation case study reproduces the split:
an external registry check reaches selection efficiency 1.00 at zero token
cost; the same model as its own critic reaches 0.00.

This gives practitioners a measurable rule for a decision currently made by
intuition, which we believe fits the applied, reproducibility-focused scope of
IEEE Access.

Every number in the paper can be re-derived without a GPU: the code, the
frozen 540-task suite and all cached generations are public at
https://github.com/madhavamshahi/generator-verifier-gap.

The manuscript is original, is not under consideration elsewhere, and all
authors have approved this submission. We declare no conflicts of interest.

Sincerely,
Madhavam Pratap Shahi, on behalf of all authors
madhavam.shahi.12@gmail.com
