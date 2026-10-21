# Grimace_Analysis
Multi-state analysis based on Grimace patterns

## Running report

All analyses feed one PDF report (`report/output/grimace_report.pdf`), organised by the research question IDs in `RESEARCH_QUESTIONS.md` (data D1-D2, Q1-Q6, A1-A4, B1-B5).

```
python report/run_all.py          # re-run all analysis scripts and rebuild the PDF
python report/run_all.py --build  # only rebuild the PDF from the saved blocks
```

Each analysis script saves a `grimace.report.Block` (question, data, method, figures, findings, interpretation, caveats, open points); see `src/grimace/report.py`.
To add an analysis: write the script, add it to `report/run_all.py`, and add a new question to `report/questions.yaml` if needed.
