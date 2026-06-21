# Outputs

Generated artifacts (not all are committed to git).

| Path | Description |
|------|-------------|
| `experiments_table.csv` | Metrics E1–E5 (committed) |
| `experiments_comparison.png` | Bar charts (committed) |
| `eda_bbox_distribution.png` | EDA histogram (committed) |
| `experiments/` | Training runs, weights (ignored) |
| `error_analysis/` | FP/FN examples (ignored) |
| `log.md` | Training journal (ignored) |

Regenerate:

```bash
python scripts/collect_results.py
python scripts/run_analysis.py
```