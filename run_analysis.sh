#!/bin/bash
# Full analysis chain: raw results -> tables, figures, compiled paper.
set -e
cd "$(dirname "$0")"
echo "== aggregate =="
python3 gvgap/analysis.py
echo "== matched-lift bootstrap =="
python3 -c "
import sys; sys.path.insert(0,'.')
import pandas as pd
from gvgap.analysis import load_raw, tidy, matched_lift_ci
df = tidy(load_raw())
out = matched_lift_ci(df)
out.to_csv('results/matched_lift_ci.csv', index=False)
print('matched_lift_ci rows:', len(out))
"
echo "== prediction =="
python3 gvgap/predict.py
echo "== report =="
python3 gvgap/report.py
echo "== figures =="
python3 gvgap/figures.py
echo "== compile =="
cd paper && tectonic -X compile main.tex 2>&1 | tail -3
echo "DONE: paper/main.pdf"
