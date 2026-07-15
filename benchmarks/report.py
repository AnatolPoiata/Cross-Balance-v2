"""Turn per-fold benchmark results into results_summary.csv/.html."""

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import METRIC_NAMES


def summarize(fold_results):
    """``fold_results``: {method_name: {metric: [per-fold values]}} ->
    a DataFrame with one row per method, mean/std per metric, ranked by AUCPR.
    """
    rows = []
    for method, metrics in fold_results.items():
        row = {"method": method}
        for name in METRIC_NAMES:
            values = np.asarray(metrics[name], dtype=float)
            row[f"{name}_mean"] = np.nanmean(values)
            row[f"{name}_std"] = np.nanstd(values)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("aucpr_mean", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def write_report(summary_df, output_dir, title="Cross-Balance Benchmark"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "results_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    display_cols = ["rank", "method"] + [f"{m}_mean" for m in METRIC_NAMES]
    display_df = summary_df[display_cols].copy()
    display_df.columns = ["Rank", "Method"] + [m.replace("_", " ").upper() for m in METRIC_NAMES]
    for col in display_df.columns[2:]:
        display_df[col] = display_df[col].map(lambda v: f"{v:.4f}")

    table_html = display_df.to_html(index=False, border=0, classes="results")
    highlight_row = summary_df.iloc[0]["method"] if len(summary_df) else "n/a"

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; color: #1a1a1a; }}
table.results {{ border-collapse: collapse; width: 100%; }}
table.results th, table.results td {{ padding: 0.4rem 0.8rem; text-align: right; border-bottom: 1px solid #ddd; }}
table.results th:nth-child(2), table.results td:nth-child(2) {{ text-align: left; }}
table.results tr:first-child {{ font-weight: bold; background: #f0f7ff; }}
h1 {{ font-size: 1.4rem; }}
p.meta {{ color: #666; }}
</style>
</head><body>
<h1>{title}</h1>
<p class="meta">Top method by AUCPR: <strong>{highlight_row}</strong></p>
{table_html}
</body></html>
"""
    html_path = output_dir / "report.html"
    html_path.write_text(html)
    return csv_path, html_path
