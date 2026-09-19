# The Generator–Verifier Gap

Code and data for *The Generator–Verifier Gap Predicts When Multi-Agent LLM
Architectures Beat a Compute-Matched Single Agent*.

Madhavam Pratap Shahi · Kavish Soningra · Nagendra Chaudhary
Independent Researchers

Everything here runs locally on Apple Silicon with open-weight models. There is
no API key, no paid inference, and no hidden cost — which matters for a paper
whose central claim is about compute accounting.

## What the paper argues

A $k$-agent ensemble's accuracy factors exactly into **coverage** (does any
candidate get it right?) and **selection efficiency** (can the selector find
it?). Selection efficiency is predictable in advance from a cheap pairwise
probe — the generator–verifier gap $G$ — via a latent-score model with no
fitted parameters. Priced against a single agent given the *same* token budget,
most apparent multi-agent gains disappear.

## Layout

```
gvgap/
  synth.py        generators for the three synthetic task families
  build_data.py   assembles the frozen 6-family task suite
  verify.py       ground-truth checkers (one per family)
  engine.py       batched MLX inference with token accounting + caching
  arch.py         architectures: voting, judging, program verifier, debate, refine
  theory.py       latent-score model linking pairwise G to listwise selection
  metrics.py      pass@k, bootstrap CIs, Holm correction
  run.py          main experiment driver (resumable)
  analysis.py     aggregation and token-matched lift
  predict.py      H2 prediction and the build/don't-build decision rule
  citations.py    OpenAlex resolution for the citation case study
  run_citations.py the case-study experiment
  report.py       emits every number and table the paper prints
  figures.py      publication figures
paper/            LaTeX source; main.pdf is the compiled preprint
```

## Reproducing

```bash
python3 gvgap/build_data.py            # frozen task suite
python3 gvgap/run.py                   # all models x families (resumable)
python3 gvgap/run_citations.py         # citation case study
./run_analysis.sh                      # tables, figures, compiled PDF
```

Generations are cached by a hash of (model, prompt, max tokens, temperature,
seed) in `results/cache/`, so an interrupted run resumes for free and a repeat
run costs nothing.

## Requirements

Python 3.9+, `mlx`, `mlx-lm`, `numpy`, `pandas`, `scipy`, `matplotlib`,
`huggingface_hub`, and `tectonic` for the PDF.

## License

MIT for the code. Benchmark data retains its original licensing
(GSM8K, MBPP, ARC).
