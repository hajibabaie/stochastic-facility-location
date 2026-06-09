# Technical report

LaTeX source for the technical report on this project.

## Build

```bash
latexmk -pdf main.tex          # or: pdflatex; bibtex; pdflatex; pdflatex
```

Output: `main.pdf`.

## Regenerating the numbers

Every number and figure comes from one script run against the repository:

```bash
uv run python scripts/collect_results.py
```

It writes `results/report/results.json` and figures under
`results/report/figures/`. The figures used by the report are copied into
`paper/figures/`, and the JSON is copied to `paper/data/results.json` for
provenance. The tables in `main.tex` are transcribed from that JSON (seed
`20231015`).
