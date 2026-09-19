"""Per-family instructions. Held fixed across every architecture.

The only thing that varies between conditions is the orchestration, never the
task framing, so that any accuracy difference is attributable to architecture
rather than to prompt engineering.
"""

SYSTEM = {
    "gsm8k": "Solve the problem step by step. End your reply with 'Answer: <number>'.",
    "csp": "Solve the problem. Show brief reasoning. End your reply with 'Answer: <integer>'.",
    "logic": "Work out the full ordering, then answer. End your reply with 'Answer: <name>'.",
    "arc": "Answer the multiple-choice question. End your reply with 'Answer: <letter>'.",
    "wordcon": "Follow the constraints exactly. Output only the finished sentence on the last line, with no quotes or commentary.",
    "mbpp": "Write a correct Python function. Output the complete function inside a ```python code block. No explanation.",
}

# Selecting the best of several candidates.
JUDGE_SYSTEM = (
    "You are evaluating candidate answers to a problem. "
    "Decide which candidate is correct. "
    "End your reply with 'Best: <number>'."
)

JUDGE_USER = """Problem:
{problem}

{candidates}

Which candidate is correct? End your reply with 'Best: <number>'."""

# Pairwise discrimination probe used to measure the generator-verifier gap.
PAIR_SYSTEM = (
    "You are checking answers to a problem. "
    "Exactly one of the two candidates is correct. "
    "End your reply with 'Correct: A' or 'Correct: B'."
)

PAIR_USER = """Problem:
{problem}

Candidate A:
{a}

Candidate B:
{b}

Exactly one candidate is correct. Which one? End your reply with 'Correct: A' or 'Correct: B'."""

REFINE_USER = """Problem:
{problem}

Your previous attempt:
{previous}

Review the attempt critically. If it contains an error, fix it. Then give your final answer in the required format."""

DEBATE_USER = """Problem:
{problem}

Here are answers from other solvers:
{others}

Considering these, give your own final answer in the required format."""

# A second single-agent way to spend a larger budget: one long chain of
# thought rather than several revision passes. Reported alongside refinement so
# that the single-agent frontier is not defined by one possibly-weak method.
LONG_SUFFIX = (
    "\n\nThink through this carefully and at length. Consider more than one "
    "approach, check your working for errors, and only then give your final "
    "answer in the required format."
)
