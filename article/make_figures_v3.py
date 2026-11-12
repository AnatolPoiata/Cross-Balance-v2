"""Figures for the v3 article (EN and RO), from results/v3 only.

    ./.venv/bin/python -m article.make_figures_v3
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmarks.analyze_v3 import LABELS, friedman_nemenyi, uncorrected  # noqa: E402
from benchmarks.run_v3 import OUT_DIR, load_cached  # noqa: E402

ANALYSIS = OUT_DIR / "analysis_article"
ULB_METHODS = ["CrossBalance_v3", "CB3_uncorrected", "XGBoost_spw", "RF_balanced", "LightGBM"]

FIG_DIR = Path(__file__).resolve().parent / "figures_v3"
BLUE, RED, GRAY, INK = "#1f5fa8", "#c0392b", "#9a9a9a", "#222222"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8, "axes.edgecolor": INK, "axes.linewidth": 0.6})

RO_LABELS = {"LR (class weights)": "RL (ponderi de clasă)", "SMOTE + LR": "SMOTE + RL",
             "RF (class weights)": "RF (ponderi de clasă)",
             "Cross-Balance (uncorrected)": "Cross-Balance (necorectat)"}


def lab(m, lang):
    s = LABELS.get(m, m)
    return RO_LABELS.get(s, s) if lang == "ro" else s


def cd_figure(lang):
    tab = pd.read_csv(ANALYSIS / "ap_holdout.csv", index_col=0)
    ranks, fr = friedman_nemenyi(tab)
    k, cd = len(ranks), fr["cd"]
    fig, ax = plt.subplots(figsize=(6.3, 1.95), dpi=300)
    lo, hi = 1, k
    ax.set_xlim(lo - 3.2, hi + 3.2)
    ax.set_ylim(0, 1)
    ax.axis("off")
    axis_y = 0.80
    ax.hlines(axis_y, lo, hi, color=INK, lw=0.8)
    for r in range(lo, hi + 1):
        ax.vlines(r, axis_y - 0.015, axis_y + 0.015, color=INK, lw=0.8)
        ax.text(r, axis_y + 0.03, str(r), ha="center", va="bottom", fontsize=7)
    ax.hlines(0.97, lo, lo + cd, color=RED, lw=1.6)
    ax.vlines([lo, lo + cd], 0.95, 0.99, color=RED, lw=1.0)
    ax.text(lo + cd + 0.15, 0.97, f"CD = {cd:.2f}".replace(".", ",") if lang == "ro" else f"CD = {cd:.2f}",
            va="center", fontsize=7, color=RED)
    half = (k + 1) // 2
    step = 0.66 / half
    for i, (m, r) in enumerate(ranks.items()):
        left = i < half
        y = 0.66 - (i if left else k - 1 - i) * step
        x_end = lo - 0.4 if left else hi + 0.4
        color = BLUE if m == "CrossBalance_v3" else INK
        ax.plot([r, r, x_end], [axis_y, y, y], color=color, lw=0.7)
        rr = f"{r:.2f}".replace(".", ",") if lang == "ro" else f"{r:.2f}"
        ax.text(x_end + (-0.08 if left else 0.08), y, f"{lab(m, lang)} ({rr})", ha="right" if left else "left",
                va="center", fontsize=7, color=color, weight="bold" if m == "CrossBalance_v3" else "normal")
    vals = ranks.values
    cliques = []
    for i in range(k):
        j = max(jj for jj in range(i, k) if vals[jj] - vals[i] <= cd)
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c, (i, j) in enumerate(cliques):
        ax.hlines(axis_y - 0.05 - c * 0.035, vals[i] - 0.05, vals[j] + 0.05, color=GRAY, lw=2.0)
    fig.tight_layout(pad=0.2)
    out = FIG_DIR / f"fig1_cd_holdout_{lang}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def ulb_curves():
    X, y = load_cached("creditcard_ulb")
    amount = X[:, -1]
    budgets = np.array([0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01])
    c_grid = np.array([1, 2, 5, 10, 20, 50])
    out = {}
    for m in ULB_METHODS:
        src = "CrossBalance_v3" if m == "CB3_uncorrected" else m
        rec_amt, sav = [], []
        for rep in range(3):
            proba = np.full(len(y), np.nan)
            for f in range(5):
                d = np.load(OUT_DIR / "preds" / f"creditcard_ulb__{src}__{rep}_{f}.npz")
                proba[d["idx"]] = (uncorrected(d["proba"].astype(float), y, d["idx"]) if m == "CB3_uncorrected"
                                   else d["proba"])
            order = np.argsort(-proba, kind="stable")
            row = []
            for b in budgets:
                top = order[: int(round(b * len(y)))]
                row.append(amount[top][y[top] == 1].sum() / amount[y == 1].sum())
            rec_amt.append(row)
            loss = amount[y == 1].sum()
            sav.append([1 - (amount[(y == 1) & ~(proba * amount > c)].sum() + c * (proba * amount > c).sum()) / loss
                        for c in c_grid])
        out[m] = (np.mean(rec_amt, axis=0), np.mean(sav, axis=0))
    return budgets, c_grid, out


def ulb_figure(lang, data):
    budgets, c_grid, out = data
    show = {"CrossBalance_v3": (BLUE, "-", 1.6), "XGBoost_spw": (RED, "-", 1.1), "RF_balanced": (INK, "--", 1.0),
            "LightGBM": (GRAY, "-.", 1.0), "CB3_uncorrected": (BLUE, ":", 1.2)}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.35), dpi=300)
    for m, (c, ls, lw) in show.items():
        if m != "CB3_uncorrected":  # identical ranking to Cross-Balance, so only shown in panel (b)
            axes[0].plot(budgets * 100, out[m][0] * 100, color=c, ls=ls, lw=lw, marker="o", ms=2.2,
                         label=lab(m, lang))
        axes[1].plot(c_grid, out[m][1] * 100, color=c, ls=ls, lw=lw, marker="o", ms=2.2, label=lab(m, lang))
    if lang == "en":
        axes[0].set_xlabel("Investigation budget (% of transactions reviewed, log scale)")
        axes[0].set_ylabel("Fraud amount in reviewed cases (%)")
        axes[1].set_xlabel("Cost of one investigation (EUR, log scale)")
        axes[1].set_ylabel("Net savings vs. no model (%)")
        axes[0].set_title("(a) Fixed review capacity", fontsize=8)
        axes[1].set_title("(b) Cost rule  p(x)·A(x) > c", fontsize=8)
    else:
        axes[0].set_xlabel("Buget de investigare (% tranzacții verificate, scară log.)")
        axes[0].set_ylabel("Suma fraudelor în cazurile verificate (%)")
        axes[1].set_xlabel("Costul unei investigații (EUR, scară log.)")
        axes[1].set_ylabel("Economii nete față de lipsa modelului (%)")
        axes[0].set_title("(a) Capacitate fixă de verificare", fontsize=8)
        axes[1].set_title("(b) Regula de cost  p(x)·A(x) > c", fontsize=8)
    axes[0].set_xscale("log")
    axes[0].set_xticks([0.05, 0.1, 0.2, 0.5, 1.0])
    axes[0].set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1"] if lang == "en" else ["0,05", "0,1", "0,2", "0,5", "1"])
    axes[0].minorticks_off()
    axes[0].axvline(0.1, color=GRAY, lw=0.6, ls=":")
    axes[1].set_xscale("log")
    axes[1].set_xticks(c_grid)
    axes[1].set_xticklabels([str(c) for c in c_grid])
    axes[1].set_ylim(-250, 100)
    axes[1].axhline(0, color=GRAY, lw=0.5)
    for ax in axes:
        ax.grid(color="#e5e5e5", lw=0.5)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=lw, label=lab(m, lang)) for m, (c, ls, lw) in show.items()]
    axes[0].legend(handles=handles, fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3, w_pad=1.2)
    path = FIG_DIR / f"fig2_ulb_economics_{lang}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    FIG_DIR.mkdir(exist_ok=True)
    data = ulb_curves()
    (FIG_DIR / "ulb_curves.json").write_text(json.dumps(
        {"budgets": data[0].tolist(), "c_grid": data[1].tolist(),
         "series": {m: [a.tolist(), b.tolist()] for m, (a, b) in data[2].items()}}, indent=1))
    for lang in ("en", "ro"):
        print(cd_figure(lang), ulb_figure(lang, data))


if __name__ == "__main__":
    main()
