"""Turn per-fold benchmark results into results_summary.csv/.html/pr_curves.png."""

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import METRIC_NAMES

# Chart chrome, from the project's dataviz reference palette (light mode).
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_AXIS = "#c3c2b7"
_BLUE = "#2a78d6"    # categorical slot 1 -- reserved for Cross-Balance
_RED = "#e34948"     # categorical slot 8 -- reserved for the strongest baseline
_OTHER_GRAY = "#c3c2b7"


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


def write_report(summary_df, output_dir, title="Cross-Balance Benchmark", pr_curve_filename=None):
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
    pr_curve_html = f'<img src="{pr_curve_filename}" alt="PR curves" style="max-width:100%">' if pr_curve_filename else ""

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
{pr_curve_html}
</body></html>
"""
    html_path = output_dir / "report.html"
    html_path.write_text(html)
    return csv_path, html_path


def plot_pr_curves(oof_true, oof_proba, summary_df, output_path, title):
    """Overlay precision-recall curves for every method (11 series folds
    down to: Cross-Balance highlighted, the single strongest baseline
    highlighted, everything else as a single muted 'other methods' series
    -- per the project's dataviz guidance, categorical color doesn't scale
    past ~4 series).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    ranked = summary_df.sort_values("aucpr_mean", ascending=False)["method"].tolist()
    strongest_baseline = next((m for m in ranked if m != "CrossBalance"), None)
    highlighted = {"CrossBalance": _BLUE, strongest_baseline: _RED}

    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)

    handles = {}
    other_handle = None
    for name in oof_true:
        y_true, y_proba = oof_true[name], oof_proba[name]
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)

        if name in highlighted:
            (line,) = ax.plot(recall, precision, color=highlighted[name], linewidth=2.0,
                               label=f"{name} (AP={ap:.3f})", zorder=3)
            handles[name] = line
        else:
            (line,) = ax.plot(recall, precision, color=_OTHER_GRAY, linewidth=0.75, alpha=0.6,
                               label="Other 9 baselines", zorder=1)
            other_handle = line

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", color=_INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Precision", color=_INK_SECONDARY, fontsize=10)
    ax.set_title(title, color=_INK_PRIMARY, fontsize=12, loc="left")
    ax.grid(True, color=_GRIDLINE, linewidth=1.0, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_AXIS)
    ax.tick_params(colors=_INK_MUTED, labelsize=9)

    # Legend order: Cross-Balance (the hero) first, then the strongest
    # baseline, then the folded "other" series -- not draw order.
    ordered = [h for h in (handles.get("CrossBalance"), handles.get(strongest_baseline), other_handle) if h]
    ax.legend(handles=ordered, loc="best", fontsize=8.5, frameon=False, labelcolor=_INK_SECONDARY)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, facecolor=_SURFACE)
    plt.close(fig)
    return output_path
