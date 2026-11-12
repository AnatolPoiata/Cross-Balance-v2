"""Analysis of the v3 run (``benchmarks.run_v3``): metrics, ranks,
statistical tests, ablation, calibration and the fraud-cost analysis.

    python -m benchmarks.analyze_v3

Writes CSV tables and figures to ``results/v3/analysis/``.
"""

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

from .real_datasets import HOLDOUT_DATASETS
from .run_v3 import ABLATIONS, MAIN_METHODS, OUT_DIR, load_cached

ANALYSIS = OUT_DIR / "analysis"
TARGET = "CrossBalance_v3"
# Two-tailed Nemenyi critical values q_0.05 (Demsar, 2006, Table 5), k = 2..20.
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
       11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489,
       19: 3.517, 20: 3.544}
LABELS = {"LR_balanced": "LR (class weights)", "SMOTE_LR": "SMOTE + LR", "RF_balanced": "RF (class weights)",
          "BalancedRF": "Balanced RF", "EasyEnsemble": "EasyEnsemble", "RUSBoost": "RUSBoost",
          "XGBoost_spw": "XGBoost (w)", "LightGBM_spw": "LightGBM (w)", "LightGBM": "LightGBM", "SPE": "Self-paced Ens.",
          "CrossBalance_v2": "Cross-Balance v2", "CrossBalance_v3": "Cross-Balance",
          "CB3_uncorrected": "Cross-Balance (uncorrected)"}


def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
               for b in range(bins) if (idx == b).any())


def prior_k(y_train, ratio=4.0, cap=3000):
    """k = beta / gamma used by CrossBalanceClassifier(calibration='prior')
    for a training fold (mirrors classifier._fit_ensemble)."""
    n_min, n_maj = int(y_train.sum()), int((y_train == 0).sum())
    chunk = min(max(1, int(round(ratio * n_min))), n_maj)
    n_min_use = n_min
    if cap is not None and n_min + chunk > cap:
        scale = cap / (n_min + chunk)
        n_min_use = max(1, int(round(n_min * scale)))
        chunk = max(1, cap - n_min_use)
    return (chunk / n_maj) / (n_min_use / n_min)


def uncorrected(proba, y_all, test_idx):
    """Raw chunk-prevalence score s from p = k s / (k s - s + 1)."""
    mask = np.ones(len(y_all), bool)
    mask[test_idx] = False
    k = prior_k(y_all[mask])
    return proba / (k + proba - proba * k)


def fold_metrics(y, proba, pred):
    k = int(y.sum())
    order = np.argsort(-proba, kind="stable")
    top1 = order[: max(1, int(round(0.01 * len(y))))]
    return dict(
        ap=average_precision_score(y, proba), roc_auc=roc_auc_score(y, proba),
        brier=brier_score_loss(y, np.clip(proba, 0, 1)), ece=ece(y, np.clip(proba, 0, 1)),
        r_prec=y[order[:k]].mean(), p_at_1pct=y[top1].mean(), r_at_1pct=y[top1].sum() / k,
        precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0), mcc=matthews_corrcoef(y, pred),
    )


def load_runs():
    rows, ys = [], {}
    for p in sorted((OUT_DIR / "preds").glob("*.json")):
        if p.name.startswith("._"):
            continue
        r = json.loads(p.read_text())
        if not r.get("ok"):
            rows.append(r)
            continue
        if r["ds"] not in ys:
            ys[r["ds"]] = load_cached(r["ds"])[1]
        d = np.load(p.with_suffix(".npz"))
        y_te = ys[r["ds"]][d["idx"]]
        r.update(fold_metrics(y_te, d["proba"].astype(float), d["pred"]))
        rows.append(r)
        if r["method"] == "CrossBalance_v3":
            raw = uncorrected(d["proba"].astype(float), ys[r["ds"]], d["idx"])
            r2 = dict(r, method="CB3_uncorrected")
            r2.update(fold_metrics(y_te, raw, (raw >= 0.5).astype(int)))
            rows.append(r2)
    return pd.DataFrame(rows)


def friedman_nemenyi(table):
    """table: datasets x methods of mean AP (higher is better)."""
    ranks = table.rank(axis=1, ascending=False)
    n, k = table.shape
    chi2, p = stats.friedmanchisquare(*[table[c] for c in table.columns])
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2)  # Iman-Davenport correction
    p_ff = 1 - stats.f.cdf(ff, k - 1, (k - 1) * (n - 1))
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * n))
    return ranks.mean().sort_values(), dict(n=n, k=k, chi2=chi2, p=p, ff=ff, p_ff=p_ff, cd=cd)


def wilcoxon_holm(table, target=TARGET):
    out = []
    for m in table.columns:
        if m == target:
            continue
        d = table[target] - table[m]
        nz = d[d.abs() > 1e-9]
        p = stats.wilcoxon(nz).pvalue if len(nz) > 0 else 1.0
        out.append(dict(baseline=m, wins=int((d > 1e-9).sum()), ties=int((d.abs() <= 1e-9).sum()),
                        losses=int((d < -1e-9).sum()), mean_diff=d.mean(), p=p))
    df = pd.DataFrame(out).sort_values("p").reset_index(drop=True)
    m = len(df)
    adj, running = [], 0.0
    for i, p in enumerate(df.p):  # Holm step-down
        running = max(running, min(1.0, (m - i) * p))
        adj.append(running)
    df["p_holm"] = adj
    return df


def cd_diagram(avg_ranks, cd, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(avg_ranks)
    fig, ax = plt.subplots(figsize=(6.3, 2.5), dpi=200)
    lo, hi = 1, int(np.ceil(avg_ranks.max()))
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.hlines(0.82, lo, hi, color="#333", lw=1)
    for r in range(lo, hi + 1):
        ax.vlines(r, 0.80, 0.84, color="#333", lw=1)
        ax.text(r, 0.87, str(r), ha="center", va="bottom", fontsize=7, family="serif")
    ax.hlines(0.97, lo, lo + cd, color="#b2182b", lw=2)
    ax.text(lo + cd / 2, 0.99, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7, family="serif",
            color="#b2182b")
    half = (k + 1) // 2
    for i, (m, r) in enumerate(avg_ranks.items()):
        left = i < half
        y = 0.70 - (i if left else k - 1 - i) * (0.62 / half)
        x_text = lo - 0.25 if left else hi + 0.25
        color = "#2166ac" if m == TARGET else "#333"
        weight = "bold" if m == TARGET else "normal"
        ax.plot([r, r, x_text], [0.82, y, y], color=color, lw=0.8)
        ax.text(x_text + (-0.05 if left else 0.05), y, f"{LABELS.get(m, m)} ({r:.2f})",
                ha="right" if left else "left", va="center", fontsize=6.5, family="serif",
                color=color, weight=weight)
    # cliques: groups of methods not significantly different
    vals = avg_ranks.values
    cliques = []
    for i in range(k):
        j = max(jj for jj in range(i, k) if vals[jj] - vals[i] <= cd)
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c, (i, j) in enumerate(cliques):
        yy = 0.77 - c * 0.035
        ax.hlines(yy, vals[i] - 0.03, vals[j] + 0.03, color="#666", lw=2.2)
    ax.set_title(title, fontsize=8, family="serif")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cost_analysis(c_invs=(1, 2, 5, 10, 20, 50), methods=None):
    """ULB fraud: example-dependent cost of a Bayes decision rule
    flag(x) <=> p(x) * amount(x) > c_inv (Elkan, 2001; Bahnsen et al., 2014),
    plus investigation-budget metrics. Savings are relative to no model
    (every fraud amount lost)."""
    X, y = load_cached("creditcard_ulb")
    amount = X[:, -1]
    rows, budget_rows = [], []
    for m in methods or MAIN_METHODS:
        src = "CrossBalance_v3" if m == "CB3_uncorrected" else m
        for rep in range(3):
            proba = np.full(len(y), np.nan)
            for f in range(5):
                p = OUT_DIR / "preds" / f"creditcard_ulb__{src}__{rep}_{f}.npz"
                if not p.exists():
                    break
                d = np.load(p)
                proba[d["idx"]] = (uncorrected(d["proba"].astype(float), y, d["idx"]) if m == "CB3_uncorrected"
                                   else d["proba"])
            if np.isnan(proba).any():
                continue
            loss_none = amount[y == 1].sum()
            for c in c_invs:
                flag = proba * amount > c
                cost = amount[(y == 1) & ~flag].sum() + c * flag.sum()
                rows.append(dict(method=m, repeat=rep, c_inv=c, alerts=int(flag.sum()),
                                 savings=1 - cost / loss_none))
            order = np.argsort(-proba, kind="stable")
            for frac in (0.001, 0.002, 0.005, 0.01):
                top = order[: int(round(frac * len(y)))]
                budget_rows.append(dict(method=m, repeat=rep, budget=frac,
                                        precision=y[top].mean(), recall=y[top].sum() / y.sum(),
                                        amount_recall=amount[top][y[top] == 1].sum() / amount[y == 1].sum()))
    cost = pd.DataFrame(rows).groupby(["method", "c_inv"]).savings.mean().unstack()
    budget = pd.DataFrame(budget_rows).groupby(["method", "budget"]).mean(numeric_only=True).drop(columns="repeat")
    return cost, budget


ARTICLE_METHODS = [m for m in MAIN_METHODS if m != "CrossBalance_v2"]
ARTICLE_ABLATIONS = [m for m in ABLATIONS if m != "CB3_f1_threshold"]


def main(article=False):
    """Default: full analysis incl. the legacy v2 configuration (results/v3/analysis).
    ``--article``: the method set reported in the paper, where Cross-Balance is
    presented as a new method (no legacy configuration), written to
    results/v3/analysis_article."""
    out = OUT_DIR / ("analysis_article" if article else "analysis")
    out.mkdir(parents=True, exist_ok=True)
    methods = ARTICLE_METHODS if article else MAIN_METHODS
    runs = load_runs()
    runs.to_csv(out / "fold_metrics.csv", index=False)
    failed = runs[runs.ok != True]  # noqa: E712
    print(f"{len(runs)} runs, {len(failed)} failed")
    if len(failed):
        print(failed.groupby(["method", "ds"]).size())
    ok = runs[runs.ok == True]  # noqa: E712
    metrics = ["ap", "roc_auc", "brier", "ece", "r_prec", "p_at_1pct", "r_at_1pct", "precision", "recall",
               "f1", "mcc", "fit_s"]
    summary = ok.groupby(["ds", "method"])[metrics].agg(["mean", "std"])
    summary.to_csv(out / "summary_mean_std.csv")
    mean = ok.groupby(["ds", "method"])[metrics].mean()

    hold = [d for d in mean.index.levels[0] if d in HOLDOUT_DATASETS]
    dev = [d for d in mean.index.levels[0] if d not in HOLDOUT_DATASETS]
    report = {}
    for group, dss in (("all", dev + hold), ("dev", dev), ("holdout", hold)):
        for metric in ("ap", "roc_auc", "r_prec"):
            tab = mean[metric].unstack()[methods].loc[dss].dropna(axis=0)
            avg_ranks, fr = friedman_nemenyi(tab)
            wil = wilcoxon_holm(tab)
            tab.to_csv(out / f"{metric}_{group}.csv")
            wil.to_csv(out / f"wilcoxon_{metric}_{group}.csv", index=False)
            report[f"{group}/{metric}"] = dict(friedman=fr, avg_rank=avg_ranks.round(3).to_dict(),
                                              mean=tab.mean().round(4).to_dict())
            if metric == "ap":
                cd_diagram(avg_ranks, fr["cd"], out / f"cd_ap_{group}.png",
                           f"Average rank by AP — {group} ({fr['n']} datasets)")
    # ablation (Cross-Balance variants only)
    abl_methods = ["CrossBalance_v3"] + ([] if article else ["CrossBalance_v2"]) + ARTICLE_ABLATIONS
    abl = mean["ap"].unstack()[[m for m in abl_methods if m in mean["ap"].unstack()]].dropna(axis=0)
    abl_tab = pd.DataFrame({"mean_ap": abl.mean(), "avg_rank": abl.rank(axis=1, ascending=False).mean()})
    abl_w = wilcoxon_holm(abl)
    abl_tab.to_csv(out / "ablation.csv")
    abl_w.to_csv(out / "ablation_wilcoxon.csv", index=False)
    # calibration & threshold metrics, averaged over all datasets
    calib = mean[["brier", "ece", "precision", "recall", "f1", "mcc", "fit_s"]].groupby(level="method").mean()
    calib.loc[[m for m in methods + ["CB3_f1_threshold", "CB3_uncorrected"] if m in calib.index]].to_csv(
        out / "calibration_threshold.csv")
    try:
        cost, budget = cost_analysis(methods=methods + ["CB3_uncorrected"])
        cost.to_csv(out / "ulb_cost_savings.csv")
        budget.to_csv(out / "ulb_budget.csv")
    except FileNotFoundError:
        pass
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    for key, val in report.items():
        print(key, {k: round(v, 4) if isinstance(v, float) else v for k, v in val["friedman"].items()})
        print("   ranks", val["avg_rank"])


def threshold_analysis():
    """F1 / MCC / precision / recall when every method uses an internally estimated
    F1-optimal threshold (benchmarks.thresholds_v3; Cross-Balance: CB3_f1_threshold).
    Writes analysis_article/threshold_f1_{summary,wilcoxon_*}.csv."""
    out = OUT_DIR / "analysis_article"
    thr_dir = OUT_DIR / "thresholds_f1"
    rows, ys = [], {}
    for m in ARTICLE_METHODS:
        src = "CB3_f1_threshold" if m == "CrossBalance_v3" else m
        for p in sorted((OUT_DIR / "preds").glob(f"*__{src}__*.npz")):
            if p.name.startswith("._"):
                continue
            ds = p.name.split("__")[0]
            rep_fold = p.stem.split("__")[2]
            if ds not in ys:
                ys[ds] = load_cached(ds)[1]
            d = np.load(p)
            y = ys[ds][d["idx"]]
            if m == "CrossBalance_v3":
                pred = d["pred"].astype(int)
            else:
                t = thr_dir / f"{ds}__{m}__{rep_fold}.json"
                if not t.exists():
                    continue
                pred = (d["proba"] >= json.loads(t.read_text())["threshold"]).astype(int)
            rows.append(dict(ds=ds, method=m, f1=f1_score(y, pred, zero_division=0), mcc=matthews_corrcoef(y, pred),
                             precision=precision_score(y, pred, zero_division=0),
                             recall=recall_score(y, pred, zero_division=0)))
    df = pd.DataFrame(rows)
    mean = df.groupby(["ds", "method"]).mean(numeric_only=True)
    hold = [d for d in mean.index.levels[0] if d in HOLDOUT_DATASETS]
    summ = {}
    for metric in ("f1", "mcc"):
        tab = mean[metric].unstack()[ARTICLE_METHODS].dropna(axis=0)
        summ[f"{metric}_mean_all"] = tab.mean()
        summ[f"{metric}_rank_all"] = tab.rank(axis=1, ascending=False).mean()
        summ[f"{metric}_rank_holdout"] = tab.loc[[d for d in tab.index if d in hold]].rank(
            axis=1, ascending=False).mean()
        wilcoxon_holm(tab.loc[[d for d in tab.index if d in hold]]).to_csv(
            out / f"threshold_f1_wilcoxon_{metric}_holdout.csv", index=False)
        wilcoxon_holm(tab).to_csv(out / f"threshold_f1_wilcoxon_{metric}_all.csv", index=False)
        summ[f"n_datasets_{metric}"] = pd.Series(len(tab), index=tab.columns)
    for k in ("precision", "recall"):
        summ[f"{k}_mean_all"] = mean[k].unstack()[ARTICLE_METHODS].mean()
    res = pd.DataFrame(summ).sort_values("f1_rank_all")
    res.to_csv(out / "threshold_f1_summary.csv")
    return res


if __name__ == "__main__":
    import sys

    if "--thresholds" in sys.argv:
        print(threshold_analysis().round(3).to_string())
    else:
        main(article="--article" in sys.argv)
