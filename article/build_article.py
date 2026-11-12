"""Build the ECOFORUM article on Cross-Balance (EN and RO .docx).

    ./.venv/bin/python -m benchmarks.analyze_v3 --article
    ./.venv/bin/python -m article.make_figures_v3
    ./.venv/bin/python -m article.build_article

The article presents Cross-Balance (library defaults, v0.2) as a new method;
the legacy configuration of the package is not part of the comparison. Every
number in the results sections is read from results/v3/analysis_article or
recomputed from the stored fold predictions -- nothing is retyped.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from docx.shared import Pt

from benchmarks.analyze_v3 import LABELS, uncorrected
from benchmarks.run_v3 import OUT_DIR, load_cached

from . import docx_builder as D
from . import text_front as F

ANALYSIS = OUT_DIR / "analysis_article"
HERE = Path(__file__).resolve().parent
BASE_DOCX = HERE / "Cross_Balance_ECOFORUM_EN.docx"  # carries the ECOFORUM styles / page set-up
FIG = HERE / "figures_v3"
CB = "CrossBalance_v3"  # internal method key of the library defaults
CB_RAW = "CB3_uncorrected"  # same model, scores before the prior correction
NEAR_BALANCED = ["credit_card_fraud_2023", "mammographic", "mushrooms", "fraud_detection_transactions"]
SENS_CSV = HERE.parent / "results" / "v3_devgrid" / "sensitivity_rK.csv"

NAMES = {
    "syn_99-1": "Synthetic 99:1", "syn_98-2": "Synthetic 98:2", "syn_95-5": "Synthetic 95:5",
    "communities_crime": "Communities & Crime", "credit_card_fraud": "Sparkov card fraud*",
    "synthetic_financial_fraud": "PaySim*", "fraud_detection_transactions": "Fraud transactions*",
    "credit_card_fraud_2023": "Card fraud 2023*", "ailerons": "Ailerons", "segment0": "segment0",
    "shuttle_2_vs_5": "shuttle-2 vs 5", "default_credit_card_clients": "Credit default (Taiwan)",
    "mammographic": "Mammographic mass", "mushrooms": "Secondary mushroom", "bank_marketing": "Bank marketing",
    "creditcard_ulb": "ULB card fraud", "rr_led": "led7digit", "rr_phoss": "PhosS", "rr_boundary": "boundary",
    "rr_cam": "cam", "rr_car_good": "car-good", "rr_compustat": "compustat", "rr_glass4": "glass4",
    "rr_mammography": "mammography", "rr_oil": "oil", "rr_winequality_red_8vs6": "winequality-red 8 vs 6",
    "rr_yeast_2vs8": "yeast-2 vs 8", "rr_abalone_17vs7_10": "abalone-17 vs 7-10", "rr_flare_f": "flare-F",
    "rr_hypo": "hypothyroid", "rr_sick": "sick", "rr_vowel0": "vowel0",
}
SHORT = {"LR_balanced": "LR-w", "SMOTE_LR": "SMOTE+LR", "RF_balanced": "RF-w", "BalancedRF": "BRF",
         "EasyEnsemble": "EasyEns.", "RUSBoost": "RUSBoost", "XGBoost_spw": "XGB-w", "LightGBM_spw": "LGBM-w",
         "LightGBM": "LGBM", "SPE": "SPE", CB: "CB"}
ABL_NAMES = {
    "en": {CB: "Cross-Balance (default)", "CB3_oof_weights": "member weights by internal OOF AP",
           "CB3_no_cap": "no chunk-size cap", "CB3_ratio2": "ratio r = 2",
           "CB3_boosting_v2": "error-doubling resampling", "CB3_hgb_only": "HistGB members only",
           "CB3_bagging": "uniform random chunks", "CB3_auto_ladder": "density-based model choice"},
    "ro": {CB: "Cross-Balance (implicit)", "CB3_oof_weights": "ponderi după AP OOF intern",
           "CB3_no_cap": "fără plafon de dimensiune", "CB3_ratio2": "raport r = 2",
           "CB3_boosting_v2": "reeșantionare cu dublarea erorilor", "CB3_hgb_only": "doar membri HistGB",
           "CB3_bagging": "eșantioane aleatoare uniforme", "CB3_auto_ladder": "model ales după densitate"},
}
RO_METHOD = {"LR (class weights)": "RL (ponderi de clasă)", "SMOTE + LR": "SMOTE + RL",
             "RF (class weights)": "RF (ponderi de clasă)", "Cross-Balance (uncorrected)": "Cross-Balance (necorectat)"}


def ml(m, lang):
    """Display name of a method in the article's language."""
    s = LABELS[m]
    return RO_METHOD.get(s, s) if lang == "ro" else s


def num(x, lang, d=3):
    s = f"{x:.{d}f}"
    return s.replace(".", ",") if lang == "ro" else s


def pint(x, lang):
    s = f"{int(round(x)):,}"
    return s.replace(",", ".") if lang == "ro" else s


def pfmt(p, lang):
    if p < 0.001:
        return "< 0,001" if lang == "ro" else "< 0.001"
    return num(p, lang, 3)


# ------------------------------------------------------------------ data
def ulb_mean_probability(method_scores="corrected"):
    y = load_cached("creditcard_ulb")[1]
    ps, ys = [], []
    for f in sorted((OUT_DIR / "preds").glob(f"creditcard_ulb__{CB}__*.npz")):
        if f.name.startswith("._"):
            continue
        d = np.load(f)
        p = d["proba"].astype(float)
        ps.append(uncorrected(p, y, d["idx"]) if method_scores == "raw" else p)
        ys.append(y[d["idx"]])
    return float(np.concatenate(ps).mean()), float(np.concatenate(ys).mean())



def ranks(tab):
    return tab.rank(axis=1, ascending=False).mean()


# ---------------------------------------------------------------- tables
def table_components(doc, lang, R=None):
    en = lang == "en"
    rows = [["Component", "Cross-Balance", "Alternative tested (Table 6)"] if en else
            ["Componentă", "Cross-Balance", "Alternativa testată (Tabelul 6)"]]
    rows += ([
        ["Majority sampling", "self-paced hardness bins on the ensemble score, eq. (2), without replacement",
         "uniform random chunks; error-doubling weights with replacement"],
        ["Members", "alternating HistGB (50 iterations) / ExtraTrees (20 trees)",
         "HistGB only; one family chosen by samples-per-feature density"],
        ["Ratio r, members K, cap C", "4, 50, 3,000", "r = 2; no cap"],
        ["Member weights", "uniform", "internal 3-fold out-of-fold AP"],
        ["Output", "prior-corrected probability, eq. (3)", "uncorrected chunk score"],
        ["Decision", "cost rule (4) or internal F1 threshold", "fixed threshold 0.5 (library default)"],
    ] if en else [
        ["Eșantionarea majoritară", "intervale de dificultate auto-ritmate pe scorul ansamblului, ec. (2), "
                                    "fără întoarcere", "eșantioane aleatoare uniforme; ponderi dublate la eroare, "
                                                       "cu întoarcere"],
        ["Membri", "HistGB (50 de iterații) / ExtraTrees (20 de arbori) alternanți",
         "doar HistGB; o familie aleasă după densitatea observații/variabile"],
        ["Raport r, membri K, plafon C", "4, 50, 3.000", "r = 2; fără plafon"],
        ["Ponderile membrilor", "uniforme", "AP out-of-fold internă (3 partiții)"],
        ["Ieșire", "probabilitate corectată a priori, ec. (3)", "scor necorectat al eșantionului"],
        ["Decizie", "regula de cost (4) sau prag F1 intern", "prag fix 0,5 (implicit în bibliotecă)"],
    ])
    D.table(doc, rows, [3.2, 6.6, 6.1], font_pt=8, align=["l", "l", "l"])


def table_datasets(doc, lang, R):
    info = R["info"]
    dev, hold = list(R["ap"]["dev"].index), list(R["ap"]["holdout"].index)
    rows = [["Development", "N", "p", "% min.", "Hold-out", "N", "p", "% min."] if lang == "en" else
            ["Dezvoltare", "N", "p", "% min.", "Control", "N", "p", "% min."]]
    for i in range(max(len(dev), len(hold))):
        row = []
        for group in (dev, hold):
            if i < len(group):
                n, p, m = info[group[i]]
                row += [NAMES[group[i]], pint(n, lang), str(p), num(m, lang, 2)]
            else:
                row += ["", "", "", ""]
        rows.append(row)
    D.table(doc, rows, [3.35, 1.25, 0.75, 1.0, 3.35, 1.25, 0.75, 1.0], font_pt=7.5,
            align=["l", "r", "r", "r", "l", "r", "r", "r"])



def table_wilcoxon(doc, lang, R):
    en = lang == "en"
    hdr = (["Cross-Balance vs.", "W / T / L hold-out", "Δ AP", "p (Holm)", "W / T / L all 32", "Δ AP",
            "p (Holm)"] if en else
           ["Cross-Balance față de", "C / E / P control", "Δ AP", "p (Holm)", "C / E / P toate 32", "Δ AP",
            "p (Holm)"])
    h, a = R["wil"]["holdout"].set_index("baseline"), R["wil"]["all"].set_index("baseline")
    rows = [hdr]
    for m in ranks(R["ap"]["holdout"]).sort_values().index:
        if m == CB:
            continue
        rows.append([ml(m, lang), f"{h.loc[m, 'wins']} / {h.loc[m, 'ties']} / {h.loc[m, 'losses']}",
                     num(h.loc[m, "mean_diff"], lang), pfmt(h.loc[m, "p_holm"], lang),
                     f"{a.loc[m, 'wins']} / {a.loc[m, 'ties']} / {a.loc[m, 'losses']}",
                     num(a.loc[m, "mean_diff"], lang), pfmt(a.loc[m, "p_holm"], lang)])
    D.table(doc, rows, [3.6, 2.2, 1.4, 1.6, 2.2, 1.4, 1.6], font_pt=7.5)


def table_holdout(doc, lang, R):
    en = lang == "en"
    t = R["ap"]["holdout"]
    hdr = (["Hold-out dataset", "CB", "RF-w", "XGB-w", "LGBM", "SPE", "Best other (AP)", "Rank CB"] if en else
           ["Set de control", "CB", "RF-w", "XGB-w", "LGBM", "SPE", "Cea mai bună altă metodă (AP)", "Rang CB"])
    rows, bold = [hdr], set()
    rk = t.rank(axis=1, ascending=False)[CB]
    others = t.drop(columns=[CB])
    for i, ds in enumerate(t.index, start=1):
        who = others.loc[ds].idxmax()
        rows.append([NAMES[ds], num(t.loc[ds, CB], lang), num(t.loc[ds, "RF_balanced"], lang),
                     num(t.loc[ds, "XGBoost_spw"], lang), num(t.loc[ds, "LightGBM"], lang),
                     num(t.loc[ds, "SPE"], lang), f"{SHORT[who]} ({num(others.loc[ds, who], lang)})",
                     num(rk[ds], lang, 1).replace(",0", "").replace(".0", "")])
        if rk[ds] <= 1.0:
            bold.add((i, 1))
    D.table(doc, rows, [3.6, 1.2, 1.2, 1.2, 1.2, 1.2, 3.3, 1.3], font_pt=7.5, bold_cells=bold)


def table_ablation(doc, lang, R):
    en = lang == "en"
    abl, w = R["abl"], R["abl_w"].set_index("baseline")
    hdr = (["Variant (one change from the default)", "Mean AP (32)", "Avg. rank", "Δ AP", "W / T / L of default",
            "p (Holm)", "Fit s"] if en else
           ["Varianta (o modificare față de implicit)", "AP medie (32)", "Rang mediu", "Δ AP", "C / E / P ale implicitei",
            "p (Holm)", "Antr. s"])
    rows = [hdr]
    base = abl.loc[CB, "mean_ap"]
    for m in abl.sort_values("avg_rank").index:
        if m == CB:
            rows.append([ABL_NAMES[lang][m], num(abl.loc[m, "mean_ap"], lang), num(abl.loc[m, "avg_rank"], lang, 2),
                         "—", "—", "—", num(R["fit"][m], lang, 1)])
            continue
        rows.append([ABL_NAMES[lang][m], num(abl.loc[m, "mean_ap"], lang), num(abl.loc[m, "avg_rank"], lang, 2),
                     num(abl.loc[m, "mean_ap"] - base, lang),
                     f"{w.loc[m, 'wins']} / {w.loc[m, 'ties']} / {w.loc[m, 'losses']}",
                     pfmt(w.loc[m, "p_holm"], lang), num(R["fit"][m], lang, 1)])
    D.table(doc, rows, [4.6, 1.8, 1.5, 1.5, 2.3, 1.6, 1.2], font_pt=7.5)


def table_ulb(doc, lang, R):
    en = lang == "en"
    b, c, u = R["budget"], R["cost"], R["ulb"]
    show = [CB, CB_RAW, "XGBoost_spw", "RF_balanced", "SPE", "BalancedRF", "LightGBM"]
    hdr = (["Method", "AP", "Brier ×10³", "Precision top 0.1%", "Recall top 0.2%", "Amount recall top 0.5%",
            "Savings c = 1", "Savings c = 10"] if en else
           ["Metoda", "AP", "Brier ×10³", "Precizia top 0,1%", "Sensibilitatea top 0,2%",
            "Suma recuperată top 0,5%", "Economii c = 1", "Economii c = 10"])
    rows, bold = [hdr], set()
    for m in show:
        rows.append([ml(m, lang), num(u.loc[m, "ap"], lang), num(1000 * u.loc[m, "brier"], lang, 2),
                     num(b.loc[(m, 0.001), "precision"], lang), num(b.loc[(m, 0.002), "recall"], lang),
                     num(b.loc[(m, 0.005), "amount_recall"], lang),
                     f"{num(100 * c.loc[m, '1'], lang, 1)}%", f"{num(100 * c.loc[m, '10'], lang, 1)}%"])
    for col in (1, 3, 4, 5, 6, 7):
        vals = [float(r[col].rstrip("%").replace(",", ".")) for r in rows[1:]]
        for i, v in enumerate(vals, start=1):
            if abs(v - max(vals)) < 1e-12:
                bold.add((i, col))
    D.table(doc, rows, [3.6, 1.1, 1.4, 1.8, 1.8, 2.0, 1.5, 1.5], font_pt=7.5, bold_cells=bold)


# ------------------------------------------------------------- narrative





def collect():
    R = {"report": json.loads((ANALYSIS / "report.json").read_text())}
    for key, stem in (("ap", "ap"), ("auc", "roc_auc"), ("rprec", "r_prec")):
        R[key] = {g: pd.read_csv(ANALYSIS / f"{stem}_{g}.csv", index_col=0) for g in ("dev", "holdout", "all")}
    R["wil"] = {g: pd.read_csv(ANALYSIS / f"wilcoxon_ap_{g}.csv") for g in ("dev", "holdout", "all")}
    R["abl"] = pd.read_csv(ANALYSIS / "ablation.csv", index_col=0)
    R["abl_w"] = pd.read_csv(ANALYSIS / "ablation_wilcoxon.csv")
    R["calib"] = pd.read_csv(ANALYSIS / "calibration_threshold.csv", index_col=0)
    R["cost"] = pd.read_csv(ANALYSIS / "ulb_cost_savings.csv", index_col=0)
    R["budget"] = pd.read_csv(ANALYSIS / "ulb_budget.csv", index_col=[0, 1])
    R["thr"] = pd.read_csv(ANALYSIS / "threshold_f1_summary.csv", index_col=0)
    R["thr_w"] = {m: pd.read_csv(ANALYSIS / f"threshold_f1_wilcoxon_{m}_holdout.csv").set_index("baseline")
                  for m in ("f1", "mcc")}
    R["sens"] = pd.read_csv(SENS_CSV, index_col=0)
    R["sens_gain"] = float(R["sens"]["mean_ap"].drop("r4_K50 (default)").max() - R["sens"].loc["r4_K50 (default)", "mean_ap"])
    thr_dir = ANALYSIS.parent / "thresholds_f1"
    R["thr_fallback"] = sum(1 for p in thr_dir.glob("*.json")
                            if not p.name.startswith("._") and not json.loads(p.read_text()).get("ok", True))
    fm = pd.read_csv(ANALYSIS / "fold_metrics.csv")
    R["failed"] = fm[fm.ok != True]  # noqa: E712
    ok = fm[fm.ok == True]  # noqa: E712
    R["ulb"] = ok[ok.ds == "creditcard_ulb"].groupby("method")[["ap", "brier", "fit_s"]].mean()
    R["fit"] = ok.groupby(["ds", "method"]).fit_s.mean().unstack().mean()
    R["ulb_meanp"], R["ulb_prev"] = ulb_mean_probability()
    R["info"] = {}
    for ds in R["ap"]["all"].index:
        X, y = load_cached(ds)
        R["info"][ds] = (len(y), X.shape[1], 100 * y.mean())
    dev = R["ap"]["dev"].drop(index=[d for d in NEAR_BALANCED if d in R["ap"]["dev"].index])
    R["dev_imb_ranks"] = ranks(dev)
    R["n_dev_imb"] = len(dev)
    return R


def table_summary(doc, lang, R):
    en = lang == "en"
    hdr = (["Method", "AP dev", "AP hold-out", "Rank AP dev", "Rank AP hold-out", "Rank ROC-AUC (32)",
            "Rank R-prec. (32)", "Brier", "F1*", "MCC*", "Fit s"] if en else
           ["Metoda", "AP dezv.", "AP control", "Rang AP dezv.", "Rang AP control", "Rang ROC-AUC (32)",
            "Rang prec. R (32)", "Brier", "F1*", "MCC*", "Antr. s"])
    rk_d, rk_h = ranks(R["ap"]["dev"]), ranks(R["ap"]["holdout"])
    rk_auc, rk_rp = ranks(R["auc"]["all"]), ranks(R["rprec"]["all"])
    thr = R["thr"]
    rows, bold = [hdr], set()
    for i, m in enumerate(rk_h.sort_values().index, start=1):
        rows.append([ml(m, lang), num(R["ap"]["dev"][m].mean(), lang), num(R["ap"]["holdout"][m].mean(), lang),
                     num(rk_d[m], lang, 2), num(rk_h[m], lang, 2), num(rk_auc[m], lang, 2), num(rk_rp[m], lang, 2),
                     num(R["calib"].loc[m, "brier"], lang), num(thr.loc[m, "f1_mean_all"], lang),
                     num(thr.loc[m, "mcc_mean_all"], lang), num(R["fit"][m], lang, 1)])
        if m == CB:
            bold |= {(i, j) for j in range(11)}
    D.table(doc, rows, [2.9, 1.1, 1.25, 1.25, 1.35, 1.45, 1.45, 1.1, 1.05, 1.05, 1.0], font_pt=7.5,
            bold_cells=bold)


def results_text(lang, R):
    en = lang == "en"
    rep, lab = R["report"], F.LABEL[lang]
    rk_h, rk_d = ranks(R["ap"]["holdout"]), ranks(R["ap"]["dev"])
    second_h = rk_h.drop(CB).idxmin()
    fr_h = rep["holdout/ap"]["friedman"]
    wh, wd = R["wil"]["holdout"].set_index("baseline"), R["wil"]["dev"].set_index("baseline")
    t_h = R["ap"]["holdout"]
    rk_cb = t_h.rank(axis=1, ascending=False)[CB]
    n_first, n_top3 = int((rk_cb <= 1).sum()), int((rk_cb <= 3).sum())
    margin = (t_h[CB] - t_h.drop(columns=[CB]).max(axis=1)).sort_values(ascending=False)
    gains, losses = list(margin.index[:4]), list(margin.index[::-1][:5])
    not_sig_dev = [m for m in wd.index if wd.loc[m, "p_holm"] > 0.05]
    cal, abl, aw = R["calib"], R["abl"], R["abl_w"].set_index("baseline")
    d = {m: abl.loc[m, "mean_ap"] - abl.loc[CB, "mean_ap"] for m in abl.index}
    u, b, c = R["ulb"], R["budget"], R["cost"]
    n_rus_fail = int(R["failed"].groupby("method").size().get("RUSBoost", 0))
    thr, tw = R["thr"], R["thr_w"]
    thr_order = thr.sort_values("f1_rank_all").index
    cb_f1_rank = int(list(thr_order).index(CB)) + 1
    best_f1 = thr.drop(index=CB)["f1_mean_all"].idxmax()
    best_mcc = thr.drop(index=CB)["mcc_mean_all"].idxmax()
    sig_f1 = int((tw["f1"].p_holm <= 0.05).sum())
    sig_mcc = int((tw["mcc"].p_holm <= 0.05).sum())
    ns_mcc = [m for m in tw["mcc"].index if tw["mcc"].loc[m, "p_holm"] > 0.05]
    thr_fallback = R["thr_fallback"]
    if best_f1 == best_mcc:
        best_desc_en = (f"the best baseline, {ml(best_f1, 'en')} (F1 {num(thr.loc[best_f1, 'f1_mean_all'], lang)}, "
                        f"MCC {num(thr.loc[best_f1, 'mcc_mean_all'], lang)})")
        best_desc_ro = (f"celei mai bune metode de referință, {ml(best_f1, 'ro')} (F1 "
                        f"{num(thr.loc[best_f1, 'f1_mean_all'], lang)}, MCC {num(thr.loc[best_f1, 'mcc_mean_all'], lang)})")
    else:
        best_desc_en = (f"{ml(best_f1, 'en')} (F1 {num(thr.loc[best_f1, 'f1_mean_all'], lang)}) and "
                        f"{ml(best_mcc, 'en')} (MCC {num(thr.loc[best_mcc, 'mcc_mean_all'], lang)})")
        best_desc_ro = (f"{ml(best_f1, 'ro')} (F1 {num(thr.loc[best_f1, 'f1_mean_all'], lang)}) și "
                        f"{ml(best_mcc, 'ro')} (MCC {num(thr.loc[best_mcc, 'mcc_mean_all'], lang)})")
    n_rev = int(round(0.001 * 284807))
    diff_txn = (b.loc[(CB, 0.001), "precision"] - b.loc[("XGBoost_spw", 0.001), "precision"]) * n_rev
    lgbm_over = [cc for cc in c.columns if c.loc["LightGBM", cc] > c.loc[CB, cc]]
    dim, n_dim = R["dev_imb_ranks"], R["n_dev_imb"]
    dim2 = dim.drop(CB).idxmin()
    triv = 1000 * R["ulb_prev"]  # Brier score of a model that always predicts 0

    def L(m):
        return ml(m, lang)

    def join(ms, word):
        names = [L(m) for m in ms]
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" {word} " + names[-1]

    def dsn(ms, word):
        names = [NAMES[m] for m in ms]
        return ", ".join(names[:-1]) + f" {word} " + names[-1]

    def pos(ds):
        n, _, pct = R["info"][ds]
        return int(round(n * pct / 100))

    blocks = [("h", "V. RESULTS" if en else "V. REZULTATE"),
              ("sub", "Overall comparison" if en else "Comparația generală")]
    if en:
        blocks += [
            ("p", f"Table 3 summarises the {len(rk_h)} methods. Cross-Balance has the best average rank by AP "
                  f"in both groups: {num(rk_d[CB], lang, 2)} on the development datasets and "
                  f"{num(rk_h[CB], lang, 2)} on the hold-out datasets, where the next method, {L(second_h)}, "
                  f"averages {num(rk_h[second_h], lang, 2)}. Its mean hold-out AP is "
                  f"{num(t_h[CB].mean(), lang)}, against {num(t_h[second_h].mean(), lang)}. The Friedman test "
                  f"rejects equal performance in every group (Iman-Davenport F = {num(fr_h['ff'], lang, 2)} on "
                  f"the hold-out group, p {pfmt(fr_h['p_ff'], lang)}). The four near-balanced control sets do not "
                  f"drive the development result: without them, Cross-Balance still ranks first on the remaining "
                  f"{n_dim} sets ({num(dim[CB], lang, 2)} against {num(dim[dim2], lang, 2)} for {L(dim2)})."),
            ("p", f"With {len(rk_h)} methods and 17 datasets the Nemenyi critical difference is "
                  f"{num(fr_h['cd'], lang, 2)} ranks (Fig. 1). Nemenyi is conservative, and by this criterion "
                  f"Cross-Balance is separated only from the methods whose average rank exceeds "
                  f"{num(rk_h[CB] + fr_h['cd'], lang, 2)}. The paired Wilcoxon tests in Table 4 are more "
                  f"informative. On the hold-out group, Cross-Balance has a higher AP than each baseline on "
                  f"between {int(wh.wins.min())} and {int(wh.wins.max())} of 17 datasets, and all {len(wh)} "
                  f"differences remain significant after Holm's correction (largest adjusted p = "
                  f"{num(wh.p_holm.max(), lang, 3)}). Over all 32 datasets every adjusted p is below 0.001, but "
                  f"this pool contains the 10 datasets used to select the configuration and is therefore "
                  f"optimistic; the hold-out comparison is the primary evidence. On the development group, the "
                  f"differences from {join(not_sig_dev, 'and')} are not significant at 5% after correction."),
        ]
    else:
        blocks += [
            ("p", f"Tabelul 3 rezumă cele {len(rk_h)} metode. Cross-Balance are cel mai bun rang mediu după AP în "
                  f"ambele grupuri: {num(rk_d[CB], lang, 2)} pe seturile de dezvoltare și {num(rk_h[CB], lang, 2)}"
                  f" pe seturile de control, unde următoarea metodă, {L(second_h)}, are "
                  f"{num(rk_h[second_h], lang, 2)}. AP medie pe seturile de control este "
                  f"{num(t_h[CB].mean(), lang)}, față de {num(t_h[second_h].mean(), lang)}. Testul Friedman "
                  f"respinge egalitatea performanțelor în fiecare grup (Iman-Davenport F = "
                  f"{num(fr_h['ff'], lang, 2)} pe grupul de control, p {pfmt(fr_h['p_ff'], lang)}). Cele patru "
                  f"seturi de control aproape echilibrate nu determină rezultatul de dezvoltare: fără ele, "
                  f"Cross-Balance rămâne primul pe celelalte {n_dim} seturi ({num(dim[CB], lang, 2)} față de "
                  f"{num(dim[dim2], lang, 2)} pentru {L(dim2)})."),
            ("p", f"Pentru {len(rk_h)} metode și 17 seturi, diferența critică Nemenyi este de "
                  f"{num(fr_h['cd'], lang, 2)} ranguri (Fig. 1). Testul Nemenyi este conservator și, după acest "
                  f"criteriu, Cross-Balance se separă doar de metodele cu rangul mediu peste "
                  f"{num(rk_h[CB] + fr_h['cd'], lang, 2)}. Testele Wilcoxon pereche din Tabelul 4 sunt mai "
                  f"informative: pe grupul de control, Cross-Balance are o AP mai mare decât fiecare metodă de "
                  f"referință pe {int(wh.wins.min())}-{int(wh.wins.max())} din 17 seturi, iar toate cele {len(wh)} "
                  f"diferențe rămân semnificative după corecția Holm (cea mai mare valoare p ajustată = "
                  f"{num(wh.p_holm.max(), lang, 3)}). Pe toate cele 32 de seturi, fiecare p ajustat este sub 0,001, "
                  f"dar acest ansamblu include cele 10 seturi folosite la alegerea configurației și este deci "
                  f"optimist; comparația pe datele de control rămâne dovada principală. Pe grupul de dezvoltare, "
                  f"diferențele față de {join(not_sig_dev, 'și')} nu sunt semnificative la 5% după corecție."),
        ]
    blocks += [
        ("tcap", f"{lab['table']} 3. " + (
            f"Mean AP, average ranks and threshold metrics of the {len(rk_h)} methods (lower rank is better), "
            f"sorted by hold-out rank" if en else
            f"AP medie, ranguri medii și indicatori la prag ai celor {len(rk_h)} metode (rang mai mic = mai bun), "
            f"ordonate după rangul pe datele de control")),
        ("table", "summary"),
        ("note", "Brier: mean over 32 datasets of the Brier score of the probability outputs. F1*, MCC*: mean over "
                 "32 datasets when every method uses its own F1-optimal threshold estimated by internal three-fold "
                 "cross-validation inside each training fold. Fit s: mean training time per fold on one CPU core "
                 "(Apple M-series)." if en else
                 "Brier: media pe 32 de seturi a scorului Brier al probabilităților. F1*, MCC*: media pe 32 de "
                 "seturi când fiecare metodă folosește propriul prag optim pentru F1, estimat prin validare "
                 "încrucișată internă cu trei partiții în fiecare partiție de antrenare. Antr. s: timpul mediu de "
                 "antrenare per partiție pe un nucleu (Apple seria M)."),
        ("fig", FIG / f"fig1_cd_holdout_{lang}.png", 15.8),
        ("fcap", f"{lab['figure']} 1 – " + (
            "Average AP ranks on the 17 hold-out datasets; bars join methods whose ranks differ by less than the "
            "Nemenyi critical difference (CD)" if en else
            "Ranguri medii după AP pe cele 17 seturi de control; barele unesc metodele ale căror ranguri diferă cu "
            "mai puțin decât diferența critică Nemenyi (CD)")),
        ("tcap", f"{lab['table']} 4. " + (
            "Wilcoxon signed-rank tests of Cross-Balance against each baseline (mean AP per dataset)" if en else
            "Testul Wilcoxon al rangurilor cu semn: Cross-Balance față de fiecare metodă (AP medie per set)")),
        ("table", "wilcoxon"),
        ("note", "W / T / L: datasets on which Cross-Balance has a higher / equal / lower AP; Δ AP: mean "
                 "difference (Cross-Balance minus baseline). The 32-dataset columns include the 10 selection "
                 "datasets." if en else
                 "C / E / P: seturi pe care Cross-Balance are AP mai mare / egală / mai mică; Δ AP: diferența medie "
                 "(Cross-Balance minus metoda de referință). Coloanele pentru 32 de seturi includ cele 10 seturi de "
                 "selecție."),
        ("sub", "Hold-out datasets" if en else "Seturile de control"),
    ]
    small = [ds for ds in losses if pos(ds) <= 70]
    if en:
        blocks.append(("p", f"Table 5 lists the hold-out results. Cross-Balance ranks first on {n_first} of 17 "
                            f"datasets and in the top three on {n_top3}. Its largest margins over the best "
                            f"competing method are on {dsn(gains, 'and')}. Three of these are among the largest "
                            f"hold-out sets, but oil has only 937 rows, so size alone does not explain the gains. "
                            f"Its five largest deficits are on {dsn(losses, 'and')}; {len(small)} of these have at "
                            f"most 70 positives ({', '.join(str(pos(x)) for x in small)}), where fold-to-fold "
                            f"variance is of the same order as the differences between methods. RUSBoost failed on "
                            f"{n_rus_fail} of {32 * 15} folds (AdaBoost aborts when its first learner is worse than "
                            f"chance). Its means use the remaining folds, which can only favour it."))
    else:
        blocks.append(("p", f"Tabelul 5 prezintă rezultatele pe datele de control. Cross-Balance este primul pe "
                            f"{n_first} din 17 seturi și în primele trei pe {n_top3}. Cele mai mari avantaje față "
                            f"de cea mai bună metodă concurentă apar pe {dsn(gains, 'și')}; trei dintre acestea sunt "
                            f"printre cele mai mari seturi de control, dar oil are doar 937 de rânduri, deci "
                            f"dimensiunea singură nu explică avantajul. Cele mai mari cinci deficite sunt pe "
                            f"{dsn(losses, 'și')}; {len(small)} dintre ele au cel mult 70 de cazuri pozitive "
                            f"({', '.join(str(pos(x)) for x in small)}), unde variația între partiții este de același "
                            f"ordin cu diferențele dintre metode. RUSBoost a eșuat pe {n_rus_fail} din {32 * 15} de "
                            f"partiții (AdaBoost se oprește când primul model este mai slab decât hazardul); mediile "
                            f"sale folosesc partițiile rămase, ceea ce nu îl poate dezavantaja."))
    blocks += [
        ("tcap", f"{lab['table']} 5. " + ("Mean AP (15 folds) on the hold-out datasets" if en else
                                          "AP medie (15 partiții) pe seturile de control")),
        ("table", "holdout"),
        ("note", "CB = Cross-Balance; RF-w = random forest with class weights; XGB-w / LGBM-w = XGBoost / "
                 "LightGBM with scale_pos_weight; LGBM = unweighted LightGBM; BRF = Balanced RF; SPE = Self-paced "
                 "Ensemble; LR-w = weighted logistic regression. Bold: rank 1." if en else
                 "CB = Cross-Balance; RF-w = pădure aleatoare cu ponderi de clasă; XGB-w / LGBM-w = XGBoost / "
                 "LightGBM cu scale_pos_weight; LGBM = LightGBM neponderat; BRF = Balanced RF; SPE = Self-paced "
                 "Ensemble; LR-w = regresie logistică ponderată. Îngroșat: rangul 1."),
        ("sub", "Ablation" if en else "Analiza componentelor (ablație)"),
    ]
    ties_cap = int(aw.loc["CB3_no_cap", "ties"])
    if en:
        blocks.append(("p", f"Table 6 changes one component at a time. The ablation is post hoc: it was run "
                            f"after the configuration had been fixed, on all 32 datasets including the hold-out "
                            f"group, and serves to explain the configuration, not to select it. Three choices "
                            f"matter. Uniform random chunks instead of hardness sampling cost "
                            f"{num(-d['CB3_bagging'], lang)} AP, choosing one model family per dataset by the "
                            f"samples-per-feature density of the chunks costs {num(-d['CB3_auto_ladder'], lang)}, and "
                            f"HistGB members only cost {num(-d['CB3_hgb_only'], lang)}. Error-doubling resampling "
                            f"with replacement, the AdaBoost-style alternative, loses "
                            f"{num(-d['CB3_boosting_v2'], lang)} AP, and ratio 2 loses {num(-d['CB3_ratio2'], lang)}; "
                            f"all five differences are significant. Two changes did not provide statistically "
                            f"significant gains. Adding member weights from internal out-of-fold AP, which is not "
                            f"part of the default, changes AP by {num(d['CB3_oof_weights'], lang)} (p (Holm) = "
                            f"{num(aw.loc['CB3_oof_weights', 'p_holm'], lang, 2)}) at more than twice the fitting "
                            f"time, and removing the chunk cap leaves AP unchanged on {ties_cap} of 32 datasets."))
    else:
        blocks.append(("p", f"Tabelul 6 modifică câte o componentă. Ablația este post-hoc: a fost rulată după "
                            f"fixarea configurației, pe toate cele 32 de seturi, inclusiv pe cele de control, și "
                            f"servește la explicarea configurației, nu la alegerea ei. Trei alegeri contează: "
                            f"eșantioanele aleatoare uniforme în locul eșantionării după dificultate costă "
                            f"{num(-d['CB3_bagging'], lang)} AP, alegerea unei singure familii de modele per set după "
                            f"densitatea observații/variabile costă {num(-d['CB3_auto_ladder'], lang)}, iar membrii "
                            f"exclusiv HistGB {num(-d['CB3_hgb_only'], lang)}. Reeșantionarea cu dublarea ponderilor "
                            f"la eroare și cu întoarcere, alternativa de tip AdaBoost, pierde "
                            f"{num(-d['CB3_boosting_v2'], lang)} AP, iar raportul 2 pierde "
                            f"{num(-d['CB3_ratio2'], lang)}; toate cele cinci diferențe sunt semnificative. Două "
                            f"modificări nu au adus câștiguri semnificative statistic: adăugarea ponderilor "
                            f"membrilor după AP out-of-fold internă, care nu face parte din varianta implicită, "
                            f"modifică AP cu {num(d['CB3_oof_weights'], lang)} (p Holm = "
                            f"{num(aw.loc['CB3_oof_weights', 'p_holm'], lang, 2)}) la un timp de antrenare de peste "
                            f"două ori mai mare, iar eliminarea plafonului lasă AP neschimbată pe {ties_cap} din 32 "
                            f"de seturi."))
    blocks += [
        ("tcap", f"{lab['table']} 6. " + ("Post-hoc ablation of Cross-Balance on the 32 datasets" if en else
                                          "Ablația post-hoc a Cross-Balance pe cele 32 de seturi")),
        ("table", "ablation"),
        ("sub", "Probabilities, thresholds and fraud economics" if en else
         "Probabilități, praguri și economia fraudei"),
    ]
    cbf1, cbmcc = thr.loc[CB, "f1_mean_all"], thr.loc[CB, "mcc_mean_all"]
    if en:
        blocks += [
            ("p", f"Averaged over the 32 datasets, the prior correction roughly halves the expected calibration "
                  f"error of the ensemble ({num(cal.loc[CB_RAW, 'ece'], lang)} before, "
                  f"{num(cal.loc[CB, 'ece'], lang)} after), and the mean Brier score "
                  f"({num(cal.loc[CB, 'brier'], lang)}) is close to XGBoost ({num(cal.loc['XGBoost_spw', 'brier'], lang)})"
                  f" and the weighted random forest ({num(cal.loc['RF_balanced', 'brier'], lang)}). Calibrated "
                  f"probabilities are cautious at a 0.5 threshold, which rarely suits a rare event (recall "
                  f"{num(cal.loc[CB, 'recall'], lang, 2)}, F1 {num(cal.loc[CB, 'f1'], lang, 2)}). To compare "
                  f"decision rules fairly, every method was given an F1-optimal threshold estimated by the same "
                  f"internal cross-validation (F1* and MCC* in Table 3). Under this common rule Cross-Balance "
                  f"reaches F1 {num(cbf1, lang)} and MCC {num(cbmcc, lang)} and ranks {cb_f1_rank} of {len(thr)} "
                  f"by F1, ahead of {best_desc_en}. The margin is not uniform, however: on the hold-out group its "
                  f"F1 is significantly higher than that of only {sig_f1} of {len(tw['f1'])} baselines and its MCC "
                  f"than that of {sig_mcc} (Wilcoxon-Holm); the differences from {join(ns_mcc, 'and')} are not "
                  f"significant. RUSBoost fell back to the 0.5 threshold on {thr_fallback} folds where its internal "
                  f"fits failed."),
            ("p", f"The ULB data (Table 7, Fig. 2) show the limits of this calibration. The ranking of "
                  f"Cross-Balance is on a par with XGBoost: AP {num(u.loc[CB, 'ap'], lang)} against "
                  f"{num(u.loc['XGBoost_spw', 'ap'], lang)} on a single dataset, without a significance test, "
                  f"and precision {num(b.loc[(CB, 0.001), 'precision'], lang)} against "
                  f"{num(b.loc[('XGBoost_spw', 0.001), 'precision'], lang)} when the top 0.1% ({n_rev}) "
                  f"transactions are reviewed, a difference of about {num(diff_txn, lang, 1)} transactions. "
                  f"XGBoost recovers more of the defrauded amount at that budget "
                  f"({num(100 * b.loc[('XGBoost_spw', 0.001), 'amount_recall'], lang, 1)}% against "
                  f"{num(100 * b.loc[(CB, 0.001), 'amount_recall'], lang, 1)}%). The probabilities, however, are "
                  f"over-shrunk: the mean corrected probability is {num(100 * R['ulb_meanp'], lang, 3)}% against a "
                  f"fraud rate of {num(100 * R['ulb_prev'], lang, 3)}%. Hardness sampling puts more hard "
                  f"negatives into each chunk than uniform undersampling would, so the uniform-sampling factor k in "
                  f"(3) overcorrects. As a result the corrected Brier score (×10³ "
                  f"{num(1000 * u.loc[CB, 'brier'], lang, 2)}) is slightly worse than the uncorrected one "
                  f"({num(1000 * u.loc[CB_RAW, 'brier'], lang, 2)}) and only modestly better than that of a "
                  f"constant prediction equal to the fraud rate ({num(triv, lang, 2)}), whereas XGBoost reaches "
                  f"{num(1000 * u.loc['XGBoost_spw', 'brier'], lang, 2)}. The Brier score is dominated by the "
                  f"under-predicted frauds, while the cost rule (4) is dominated by false alerts: uncorrected scores "
                  f"flag so many legitimate transactions that savings fall to "
                  f"{num(100 * c.loc[CB_RAW, '1'], lang, 0)}% at c = 1 EUR, against "
                  f"{num(100 * c.loc[CB, '1'], lang, 1)}% after correction. Better-calibrated models save more: "
                  f"XGBoost {num(100 * c.loc['XGBoost_spw', '1'], lang, 1)}% and the weighted forest "
                  f"{num(100 * c.loc['RF_balanced', '1'], lang, 1)}%. From c = {lgbm_over[0]} EUR upwards even "
                  f"unweighted LightGBM, whose AP is only {num(u.loc['LightGBM', 'ap'], lang)}, saves more than "
                  f"Cross-Balance, which confirms that savings under a cost rule depend on calibration rather than "
                  f"on ranking quality."),
        ]
    else:
        blocks += [
            ("p", f"În medie pe cele 32 de seturi, corecția a priori reduce aproximativ la jumătate eroarea de "
                  f"calibrare așteptată a ansamblului ({num(cal.loc[CB_RAW, 'ece'], lang)} înainte, "
                  f"{num(cal.loc[CB, 'ece'], lang)} după), iar scorul Brier mediu ({num(cal.loc[CB, 'brier'], lang)}) "
                  f"este apropiat de XGBoost ({num(cal.loc['XGBoost_spw', 'brier'], lang)}) și de pădurea aleatoare "
                  f"ponderată ({num(cal.loc['RF_balanced', 'brier'], lang)}). Probabilitățile calibrate sunt "
                  f"prudente la pragul 0,5, rareori potrivit pentru un eveniment rar (sensibilitate "
                  f"{num(cal.loc[CB, 'recall'], lang, 2)}, F1 {num(cal.loc[CB, 'f1'], lang, 2)}). Pentru o comparație "
                  f"echitabilă a regulilor de decizie, fiecare metodă a primit un prag optim pentru F1, estimat prin "
                  f"aceeași validare încrucișată internă (F1* și MCC* în Tabelul 3). Cu această regulă comună, "
                  f"Cross-Balance atinge F1 {num(cbf1, lang)} și MCC {num(cbmcc, lang)} și ocupă locul {cb_f1_rank} "
                  f"din {len(thr)} după F1, înaintea {best_desc_ro}. Avantajul nu este însă uniform: pe grupul de "
                  f"control, F1 este semnificativ mai mare doar decât la {sig_f1} din {len(tw['f1'])} metode de "
                  f"referință, iar MCC decât la {sig_mcc} (Wilcoxon-Holm); diferențele față de "
                  f"{join(ns_mcc, 'și')} nu sunt semnificative. RUSBoost a revenit la pragul 0,5 pe {thr_fallback} "
                  f"partiții în care antrenările sale interne au eșuat."),
            ("p", f"Datele ULB (Tabelul 7, Fig. 2) arată limitele acestei calibrări. Ordonarea Cross-Balance este "
                  f"la nivelul XGBoost: AP {num(u.loc[CB, 'ap'], lang)} față de "
                  f"{num(u.loc['XGBoost_spw', 'ap'], lang)} pe un singur set, fără test de semnificație, și precizie "
                  f"{num(b.loc[(CB, 0.001), 'precision'], lang)} față de "
                  f"{num(b.loc[('XGBoost_spw', 0.001), 'precision'], lang)} la verificarea primelor 0,1% ({n_rev}) "
                  f"tranzacții, o diferență de aproximativ {num(diff_txn, lang, 1)} tranzacții. XGBoost recuperează "
                  f"o parte mai mare din suma fraudată la acest buget "
                  f"({num(100 * b.loc[('XGBoost_spw', 0.001), 'amount_recall'], lang, 1)}% față de "
                  f"{num(100 * b.loc[(CB, 0.001), 'amount_recall'], lang, 1)}%). Probabilitățile sunt însă prea "
                  f"comprimate: probabilitatea medie corectată este de {num(100 * R['ulb_meanp'], lang, 3)}%, față "
                  f"de o rată a fraudei de {num(100 * R['ulb_prev'], lang, 3)}%. Eșantionarea după dificultate "
                  f"introduce în fiecare eșantion mai multe cazuri negative dificile decât subeșantionarea uniformă, "
                  f"astfel încât factorul k din (3), derivat pentru eșantionarea uniformă, corectează excesiv. "
                  f"Drept urmare, scorul Brier corectat (×10³ {num(1000 * u.loc[CB, 'brier'], lang, 2)}) este "
                  f"puțin mai slab decât cel necorectat ({num(1000 * u.loc[CB_RAW, 'brier'], lang, 2)}) și doar "
                  f"moderat mai bun decât al unui model care prezice mereu 0 "
                  f"({num(triv, lang, 2)}), în timp ce XGBoost atinge {num(1000 * u.loc['XGBoost_spw', 'brier'], lang, 2)}"
                  f". Scorul Brier este dominat de fraudele subestimate, iar regula de cost (4) de alertele false: "
                  f"scorurile necorectate semnalează atâtea tranzacții legitime încât economiile scad la "
                  f"{num(100 * c.loc[CB_RAW, '1'], lang, 0)}% la c = 1 EUR, față de "
                  f"{num(100 * c.loc[CB, '1'], lang, 1)}% după corecție. Modelele mai bine calibrate economisesc mai "
                  f"mult: XGBoost {num(100 * c.loc['XGBoost_spw', '1'], lang, 1)}% și pădurea ponderată "
                  f"{num(100 * c.loc['RF_balanced', '1'], lang, 1)}%. De la c = {lgbm_over[0]} EUR în sus, chiar și "
                  f"LightGBM neponderat, cu o AP de doar {num(u.loc['LightGBM', 'ap'], lang)}, economisește mai mult "
                  f"decât Cross-Balance, ceea ce confirmă că economiile într-o regulă de cost depind de calibrare, "
                  f"nu de calitatea ordonării."),
        ]
    blocks += [
        ("tcap", f"{lab['table']} 7. " + (
            "ULB card fraud (284,807 transactions, 0.17% fraud): ranking, review budget and cost-rule savings" if en
            else "Frauda cu carduri ULB (284.807 tranzacții, 0,17% fraude): ordonare, buget de verificare și "
                 "economii")),
        ("table", "ulb"),
        ("note", "Top q%: the q% highest-scored transactions are reviewed. Savings: 1 − (missed fraud amount + c × "
                 "alerts) / total fraud amount, rule (4), c in EUR per investigation. Uncorrected: the same "
                 "ensemble without eq. (3). Mean of 3 repeats of 5-fold out-of-fold predictions." if en else
                 "Top q%: sunt verificate q% dintre tranzacțiile cu cele mai mari scoruri. Economii: 1 − (suma "
                 "fraudelor ratate + c × alerte) / suma totală a fraudelor, regula (4), c în EUR per investigație. "
                 "Necorectat: același ansamblu fără ec. (3). Media a 3 repetări ale predicțiilor out-of-fold cu 5 "
                 "partiții."),
        ("fig", FIG / f"fig2_ulb_economics_{lang}.png", 15.8),
        ("fcap", f"{lab['figure']} 2 – " + (
            "ULB card fraud: share of the defrauded amount found at a fixed review budget (a; dotted line: the 0.1% "
            "budget of Table 7) and net savings of the cost rule as the investigation cost grows (b)" if en else
            "Frauda cu carduri ULB: ponderea sumei fraudate găsite la un buget fix de verificare (a; linia punctată: "
            "bugetul de 0,1% din Tabelul 7) și economiile nete ale regulii de cost pe măsură ce crește costul "
            "investigației (b)")),
    ]
    return blocks


def discussion_text(lang, R):
    fit = R["fit"]
    if lang == "en":
        return [
            ("h", "VI. DISCUSSION AND LIMITATIONS"),
            ("p", "On the hold-out datasets that played no role in its development, Cross-Balance achieved "
                  "significantly better AP than each tested default-configured baseline in paired comparisons, "
                  "and on the development data it was at least on a par with gradient boosting and weighted "
                  "forests. The post-hoc ablation attributes the gain to three choices: hardness sampling that "
                  "cannot be dominated by a few hard cases, two member families, and a larger majority share per "
                  "chunk. A second lesson concerns probabilities. Ranking quality and decision quality are "
                  "different properties: the same ensemble loses money or saves three quarters of the fraud loss "
                  "depending only on the prior correction, and a well-ranked but over-shrunk model still saves "
                  "less than better-calibrated competitors."),
            ("p", f"Several limits apply. Only Cross-Balance went through a configuration search; the baselines "
                  f"ran with default or standard settings, so tuned XGBoost or LightGBM could narrow the gap, and "
                  f"the development results show that the gap to them is small. The selected values r = 4 and "
                  f"K = 50 lie at the upper edge of the search grid; a later check on the same selection datasets "
                  f"(Appendix A) found gains of at most {num(R['sens_gain'], lang)} AP from larger values. Cross-Balance is slower than boosting "
                  f"libraries ({num(fit[CB], lang, 1)} s against {num(fit['XGBoost_spw'], lang, 1)} s per fit on "
                  f"average), although faster than a 300-tree weighted forest. Its prior correction assumes "
                  f"uniform undersampling and over-shrinks scores under hardness sampling; a sampling-aware "
                  f"correction, or isotonic or Platt calibration on internal out-of-fold scores, is the obvious "
                  f"next step. The evaluation uses random stratified folds. Fraud patterns drift, so an "
                  f"out-of-time split, which is possible on ULB, and the full-size PaySim and Sparkov sets should "
                  f"be examined before any deployment claim. Several development datasets are simulated (marked * "
                  f"in Table 2), and five reach AP near 1, so they separate methods poorly. Finally, the direct "
                  f"economic validation concerns fraud detection. For credit-risk screening the evidence is "
                  f"indirect: one credit-default dataset in the development group and general imbalanced "
                  f"benchmarks in the hold-out group, so an independent credit-risk validation is still needed."),
            ("p", "For a fraud or credit-risk unit, the practical reading is as follows. If review capacity is "
                  "fixed, Cross-Balance ranks the queue at least as well as the best alternative tested. If alerts "
                  "are raised by an expected-cost rule, a better-calibrated model, or Cross-Balance followed by an "
                  "explicit calibration step, should be preferred."),
        ]
    return [
        ("h", "VI. DISCUȚII ȘI LIMITE"),
        ("p", "Pe seturile de control, care nu au avut niciun rol în dezvoltarea metodei, Cross-Balance a obținut "
              "o AP semnificativ mai bună decât fiecare metodă de referință testată, cu setări implicite, în "
              "comparațiile pereche, iar pe datele de dezvoltare a fost cel puțin la nivelul gradient boosting-ului "
              "și al pădurilor ponderate. Ablația post-hoc atribuie câștigul a trei alegeri: o eșantionare după "
              "dificultate care nu poate fi dominată de câteva cazuri dificile, două familii de membri și o pondere "
              "majoritară mai mare în fiecare eșantion. A doua lecție privește probabilitățile: calitatea "
              "ordonării și calitatea deciziilor sunt proprietăți diferite; același ansamblu pierde bani sau "
              "economisește trei sferturi din pierderea prin fraudă doar în funcție de corecția a priori, iar un "
              "model bine ordonat, dar cu probabilități prea comprimate, economisește totuși mai puțin decât "
              "concurenții mai bine calibrați."),
        ("p", f"Există mai multe limite. Doar Cross-Balance a trecut printr-o căutare a configurației; metodele de "
              f"referință au rulat cu setări implicite sau standard, astfel încât XGBoost sau LightGBM calibrate ar "
              f"putea reduce diferența, iar rezultatele de dezvoltare arată că aceasta este mică. Valorile alese, "
              f"r = 4 și K = 50, se află la marginea superioară a grilei de căutare; o verificare ulterioară pe "
              f"aceleași seturi de selecție (Anexa A) a găsit câștiguri de cel mult {num(R['sens_gain'], lang)} AP pentru "
              f"valori mai mari. "
              f"Cross-Balance este mai lent decât bibliotecile de boosting ({num(fit[CB], lang, 1)} s față de "
              f"{num(fit['XGBoost_spw'], lang, 1)} s per antrenare, în medie), deși mai rapid decât o pădure "
              f"ponderată cu 300 de arbori. Corecția a priori presupune o subeșantionare uniformă și comprimă "
              f"excesiv scorurile în cazul eșantionării după dificultate; o corecție care ține cont de schema de "
              f"eșantionare sau calibrarea izotonică ori Platt pe scoruri interne out-of-fold este următorul pas "
              f"evident. Evaluarea folosește partiții stratificate aleatoare: tiparele de fraudă se schimbă în timp, "
              f"astfel încât o validare temporală (posibilă pe ULB) și seturile PaySim și Sparkov la dimensiune "
              f"completă trebuie examinate înaintea oricărei afirmații privind implementarea. Mai multe seturi de "
              f"dezvoltare sunt simulate (marcate cu * în Tabelul 2), iar cinci ating AP aproape de 1 și "
              f"diferențiază slab metodele. În fine, validarea economică directă privește detectarea fraudei; "
              f"pentru evaluarea riscului de credit dovezile sunt indirecte (un set de neplată a creditelor în "
              f"grupul de dezvoltare și seturi generale dezechilibrate în grupul de control), astfel încât o "
              f"validare independentă pe date de risc de credit rămâne necesară."),
        ("p", "Pentru o unitate antifraudă sau de risc de credit, concluzia practică este următoarea: dacă "
              "capacitatea de verificare este fixă, Cross-Balance ordonează lista de cazuri cel puțin la fel de bine "
              "ca cea mai bună alternativă testată; dacă alertele sunt generate printr-o regulă a costului "
              "așteptat, trebuie preferat un model mai bine calibrat sau Cross-Balance urmat de un pas explicit de "
              "calibrare."),
    ]


def conclusion_text(lang, R):
    rk_h = ranks(R["ap"]["holdout"])
    k = len(rk_h)
    if lang == "en":
        return [
            ("h", "VII. CONCLUSION"),
            ("p", f"Cross-Balance combines self-paced hardness sampling, heterogeneous members and a prior "
                  f"correction in an undersampling ensemble for rare-event screening. Evaluated once on 17 "
                  f"datasets that played no part in its design, it obtains the best average AP rank "
                  f"({num(rk_h[CB], lang, 2)}) among {k} methods and achieves significantly higher AP than each "
                  f"tested default-configured baseline in the paired hold-out comparisons, including XGBoost, "
                  f"LightGBM, Self-paced Ensemble and a weighted random forest. For fraud screening, its advantage "
                  f"lies in ranking cases for a fixed review capacity. When decisions follow an expected-cost rule, "
                  f"calibration quality, not ranking, determines savings, and there Cross-Balance trails the "
                  f"best-calibrated boosting models. Future work will add a sampling-aware calibration, out-of-time "
                  f"validation, an independent credit-risk dataset and equal-budget hyper-parameter tuning for all "
                  f"methods."),
        ]
    return [
        ("h", "VII. CONCLUZII"),
        ("p", f"Cross-Balance combină, într-un ansamblu cu subeșantionare pentru screeningul evenimentelor rare, "
              f"eșantionarea auto-ritmată după dificultate, membri eterogeni și corecția a priori. Evaluată o "
              f"singură dată pe 17 seturi care nu au contribuit la proiectare, metoda obține cel mai bun rang mediu "
              f"după AP ({num(rk_h[CB], lang, 2)}) dintre {k} metode și o AP semnificativ mai mare decât fiecare "
              f"metodă de referință testată, cu setări implicite, în comparațiile pereche pe datele de control, "
              f"inclusiv XGBoost, LightGBM, Self-paced Ensemble și o pădure aleatoare ponderată. În screeningul "
              f"fraudei, avantajul său constă în ordonarea cazurilor pentru o capacitate fixă de verificare; când "
              f"deciziile urmează o regulă a costului așteptat, economiile depind de calitatea calibrării, nu de "
              f"ordonare, iar aici Cross-Balance rămâne în urma celor mai bine calibrate modele de boosting. "
              f"Etapele următoare sunt o calibrare adaptată schemei de eșantionare, validarea temporală, un set "
              f"independent de risc de credit și calibrarea hiperparametrilor cu buget egal pentru toate metodele."),
    ]


def appendix_text(lang, R):
    s = R["sens"]
    base = s.loc["r4_K50 (default)"]
    alt = s.drop(index="r4_K50 (default)")
    best_alt = alt["mean_ap"].idxmax()
    gain = alt.loc[best_alt, "mean_ap"] - base["mean_ap"]
    lbl = {"r5_K50": "r = 5", "r6_K50": "r = 6", "r8_K50": "r = 8", "r4_K75": "K = 75", "r4_K100": "K = 100"}
    if lang == "en":
        listing = "; ".join(f"{lbl[i]}: {num(alt.loc[i, 'mean_ap'], lang)}" for i in lbl)
        return [
            ("h", "APPENDIX A. CONFIGURATION SEARCH"),
            ("p", "The configuration was selected on a predefined subset of 10 of the 15 development datasets (the "
                  "three synthetic sets, Sparkov, PaySim, Ailerons, Communities & Crime, Bank marketing, Credit "
                  "default and Mammographic mass), with three-fold cross-validation, seed 0 and AP as the "
                  "criterion. The first grid crossed four sampling schemes (uniform chunks, error-doubling "
                  "resampling, ensemble-score-weighted resampling and self-paced bins) with three member types "
                  "(random forest, extra trees, HistGB). Further grids varied the ratio (1 to 4), the cap (3,000 or "
                  "10,000), the number of members (20 or 50), bootstrapping of the minority class, mixed members, "
                  "and member weights estimated on the majority cases left out of each chunk (out-of-chunk "
                  "validation, distinct from the internal out-of-fold weighting of Table 6). Self-paced sampling "
                  "with alternating HistGB and ExtraTrees members at r = 4 and K = 50 had the best average rank and "
                  "was fixed before any hold-out dataset was loaded."),
            ("p", f"Because r = 4 and K = 50 were the largest values in the grid, a later check on the same 10 "
                  f"datasets and protocol extended both. Mean AP was {num(base['mean_ap'], lang)} for the default "
                  f"and {listing}. All five extensions rank above the default on average, but the largest gain "
                  f"({lbl[best_alt]}) is only {num(gain, lang)} AP, well below the fold-to-fold variation, so the "
                  f"default was kept; this check was run after the hold-out evaluation and did not change the "
                  f"configuration."),
        ]
    listing = "; ".join(f"{lbl[i]}: {num(alt.loc[i, 'mean_ap'], lang)}" for i in lbl)
    return [
        ("h", "ANEXA A. CĂUTAREA CONFIGURAȚIEI"),
        ("p", "Configurația a fost aleasă pe o submulțime predefinită de 10 din cele 15 seturi de dezvoltare (cele "
              "trei seturi sintetice, Sparkov, PaySim, Ailerons, Communities & Crime, Bank marketing, Credit default "
              "și Mammographic mass), prin validare încrucișată cu trei partiții, seed 0 și AP drept criteriu. Prima "
              "grilă a combinat patru scheme de eșantionare (eșantioane uniforme, reeșantionare cu dublarea "
              "erorilor, reeșantionare ponderată după scorul ansamblului și intervale auto-ritmate) cu trei tipuri "
              "de membri (random forest, extra trees, HistGB); grilele următoare au variat raportul (1-4), plafonul "
              "(3.000 sau 10.000), numărul de membri (20 sau 50), bootstrap-ul clasei minoritare, membrii micști și "
              "ponderile membrilor estimate pe cazurile majoritare rămase în afara fiecărui eșantion (validare "
              "în afara eșantionului, distinctă de ponderarea out-of-fold internă din Tabelul 6). Eșantionarea "
              "auto-ritmată cu membri HistGB și ExtraTrees alternanți la r = 4 și K = 50 a avut cel mai bun rang "
              "mediu și a fost fixată înainte de încărcarea oricărui set de control."),
        ("p", f"Deoarece r = 4 și K = 50 erau cele mai mari valori din grilă, o verificare ulterioară pe aceleași "
              f"10 seturi și cu același protocol le-a extins. AP medie a fost {num(base['mean_ap'], lang)} pentru "
              f"varianta implicită și {listing}. Toate cele cinci extensii au un rang mediu mai bun decât "
              f"varianta implicită, dar cel mai mare câștig ({lbl[best_alt]}) este de doar {num(gain, lang)} AP, mult "
              f"sub variația între partiții, astfel încât varianta implicită a fost păstrată; verificarea a fost "
              f"rulată după evaluarea pe datele de control și nu a modificat configurația."),
    ]


def abstract_text(lang, R):
    rk_h = ranks(R["ap"]["holdout"])
    second = rk_h.drop(CB).idxmin()
    if lang == "en":
        return (f"Rare events such as card fraud and loan default make classification imbalanced. Undersampling "
                f"ensembles are a common remedy, but hard-example re-weighting can let noisy cases dominate and "
                f"their scores overstate the minority probability. This paper proposes Cross-Balance, an "
                f"undersampling ensemble that combines self-paced hardness sampling, alternating gradient-boosting "
                f"and extra-trees members, and a prior correction for the probability distortion caused by "
                f"undersampling. Its configuration was selected on a predefined subset of the development data and "
                f"then evaluated once on 17 untouched hold-out datasets against ten default-configured baselines, "
                f"including XGBoost, LightGBM and Self-paced Ensemble, with repeated cross-validation and Friedman, "
                f"Nemenyi and Wilcoxon-Holm tests. On the hold-out data it has the best average rank by average "
                f"precision ({rk_h[CB]:.2f} against {rk_h[second]:.2f} for the next method) and significantly "
                f"higher average precision than every baseline in paired Wilcoxon tests after Holm correction. On "
                f"284,807 European card transactions its ranking is on a par with XGBoost, but better-calibrated "
                f"boosting models save more under an expected-cost decision rule, which shows that calibration, not "
                f"ranking alone, drives savings. The method is best suited to ranking cases for a fixed "
                f"investigation capacity.")
    return (f"Frauda cu carduri și neplata creditelor sunt evenimente rare, care fac clasificarea dezechilibrată. "
            f"Ansamblurile cu subeșantionare sunt frecvent folosite, însă reponderarea cazurilor dificile poate "
            f"lăsa câteva cazuri zgomotoase să domine, iar scorurile lor supraestimează probabilitatea clasei "
            f"minoritare. Articolul propune Cross-Balance, un ansamblu cu subeșantionare care combină eșantionarea "
            f"auto-ritmată după dificultate, membri alternanți gradient boosting și extra-trees și o "
            f"corecție a priori a distorsionării probabilităților produse de subeșantionare. Configurația a fost "
            f"aleasă pe o submulțime predefinită a datelor de dezvoltare și evaluată o singură dată pe 17 seturi "
            f"de control neatinse, față de zece metode de referință cu setări implicite, inclusiv XGBoost, "
            f"LightGBM și Self-paced Ensemble, prin validare încrucișată repetată și testele Friedman, Nemenyi și "
            f"Wilcoxon-Holm. Pe datele de control are cel mai bun rang mediu după precizia medie "
            f"({num(rk_h[CB], 'ro', 2)} față de {num(rk_h[second], 'ro', 2)}) și o precizie "
            f"medie semnificativ mai mare decât fiecare metodă de referință în testele pereche Wilcoxon-Holm. Pe 284.807 tranzacții europene cu carduri, ordonarea sa este la nivelul XGBoost, dar modelele de "
            f"boosting mai bine calibrate economisesc mai mult cu o regulă a costului așteptat, deci economiile "
            f"depind de calibrare, nu doar de ordonare. Metoda este cea mai potrivită pentru ordonarea "
            f"cazurilor la o capacitate fixă de investigare.")


REFERENCES = [
    ("*** (n.d.a) Credit card transactions fraud detection dataset (simulated with Sparkov), Kaggle, "
     "https://www.kaggle.com/datasets/kartik2112/fraud-detection, accessed 2026.", None),
    ("*** (n.d.b) Credit card fraud detection dataset 2023, Kaggle, "
     "https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023, accessed 2026.",
     None),
    ("*** (n.d.c) Fraud detection transactions dataset, Kaggle, "
     "https://www.kaggle.com/datasets/samayashar/fraud-detection-transactions-dataset, accessed 2026.", None),
    ("*** (n.d.d) Sparkov data generation: synthetic credit card transaction generator, GitHub, "
     "https://github.com/namebrandon/Sparkov_Data_Generation, accessed 2026.", None),
    ("*** (n.d.e) Datasets-ImbalancedRegression: a collection of datasets for imbalanced regression tasks, "
     "GitHub, https://github.com/antoniopedropi/Datasets-ImbalancedRegression, accessed 2026.", None),
    ("Alcalá-Fdez, J., Fernández, A., Luengo, J., Derrac, J., García, S., Sánchez, L., Herrera, F. (2011) KEEL "
     "data-mining software tool: data set repository, integration of algorithms and experimental analysis "
     "framework, Journal of Multiple-Valued Logic and Soft Computing, 17(2-3), pp.255-287.", None),
    ("Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P. (2002) SMOTE: synthetic minority over-sampling "
     "technique, Journal of Artificial Intelligence Research, 16, pp.321-357.", "https://doi.org/10.1613/jair.953"),
    ("Chen, C., Liaw, A., Breiman, L. (2004) Using random forest to learn imbalanced data, Technical Report 666, "
     "Department of Statistics, University of California, Berkeley.", None),
    ("Chen, T., Guestrin, C. (2016) XGBoost: a scalable tree boosting system, Proceedings of the 22nd ACM SIGKDD "
     "International Conference on Knowledge Discovery and Data Mining, pp.785-794.",
     "https://doi.org/10.1145/2939672.2939785"),
    ("Correa Bahnsen, A., Aouada, D., Stojanovic, A., Ottersten, B. (2016) Feature engineering strategies for "
     "credit card fraud detection, Expert Systems with Applications, 51, pp.134-142.",
     "https://doi.org/10.1016/j.eswa.2015.12.030"),
    ("Dal Pozzolo, A., Caelen, O., Johnson, R.A., Bontempi, G. (2015) Calibrating probability with undersampling "
     "for unbalanced classification, 2015 IEEE Symposium Series on Computational Intelligence, pp.159-166.",
     "https://doi.org/10.1109/SSCI.2015.33"),
    ("Dal Pozzolo, A., Caelen, O., Le Borgne, Y.-A., Waterschoot, S., Bontempi, G. (2014) Learned lessons in "
     "credit card fraud detection from a practitioner perspective, Expert Systems with Applications, 41(10), "
     "pp.4915-4928.", "https://doi.org/10.1016/j.eswa.2014.02.026"),
    ("Demšar, J. (2006) Statistical comparisons of classifiers over multiple data sets, Journal of Machine "
     "Learning Research, 7, pp.1-30.", "https://www.jmlr.org/papers/v7/demsar06a.html"),
    ("Elkan, C. (2001) The foundations of cost-sensitive learning, Proceedings of the 17th International Joint "
     "Conference on Artificial Intelligence, pp.973-978.", None),
    ("Elter, M., Schulz-Wendtland, R., Wittenberg, T. (2007) The prediction of breast cancer biopsy outcomes "
     "using two CAD approaches that both emphasize an intelligible decision process, Medical Physics, 34(11), "
     "pp.4164-4172.", "https://doi.org/10.1118/1.2786864"),
    ("Fernández, A., García, S., Galar, M., Prati, R.C., Krawczyk, B., Herrera, F. (2018) Learning from Imbalanced "
     "Data Sets, Springer, Cham.", "https://doi.org/10.1007/978-3-319-98074-4"),
    ("Geurts, P., Ernst, D., Wehenkel, L. (2006) Extremely randomized trees, Machine Learning, 63(1), pp.3-42.",
     "https://doi.org/10.1007/s10994-006-6226-1"),
    ("Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022) Why do tree-based models still outperform deep learning "
     "on typical tabular data?, Advances in Neural Information Processing Systems, 35, pp.507-520.", None),
    ("He, H., Garcia, E.A. (2009) Learning from imbalanced data, IEEE Transactions on Knowledge and Data "
     "Engineering, 21(9), pp.1263-1284.", "https://doi.org/10.1109/TKDE.2008.239"),
    ("Hernández Aros, L., Bustamante Molano, L.X., Gutiérrez-Portela, F., Moreno Hernández, J.J., Rodríguez "
     "Barrero, M.S. (2024) Financial fraud detection through the application of machine learning techniques: a "
     "literature review, Humanities and Social Sciences Communications, 11, 1130.",
     "https://doi.org/10.1057/s41599-024-03606-0"),
    ("Holm, S. (1979) A simple sequentially rejective multiple test procedure, Scandinavian Journal of "
     "Statistics, 6(2), pp.65-70.", None),
    ("Iman, R.L., Davenport, J.M. (1980) Approximations of the critical region of the Friedman statistic, "
     "Communications in Statistics - Theory and Methods, 9(6), pp.571-595.",
     "https://doi.org/10.1080/03610928008827904"),
    ("Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y. (2017) LightGBM: a highly "
     "efficient gradient boosting decision tree, Advances in Neural Information Processing Systems, 30, "
     "pp.3146-3154.", None),
    ("Kelly, M., Longjohn, R., Nottingham, K. (n.d.) The UCI Machine Learning Repository, "
     "https://archive.ics.uci.edu, accessed 2026.", None),
    ("Krawczyk, B. (2016) Learning from imbalanced data: open challenges and future directions, Progress in "
     "Artificial Intelligence, 5(4), pp.221-232.", "https://doi.org/10.1007/s13748-016-0094-0"),
    ("Liu, X.-Y., Wu, J., Zhou, Z.-H. (2009) Exploratory undersampling for class-imbalance learning, IEEE "
     "Transactions on Systems, Man, and Cybernetics, Part B, 39(2), pp.539-550.",
     "https://doi.org/10.1109/TSMCB.2008.2007853"),
    ("Liu, Z., Cao, W., Gao, Z., Bian, J., Chen, H., Chang, Y., Liu, T.-Y. (2020a) Self-paced ensemble for highly "
     "imbalanced massive data classification, 2020 IEEE 36th International Conference on Data Engineering "
     "(ICDE), pp.841-852.", "https://doi.org/10.1109/ICDE48307.2020.00078"),
    ("Liu, Z., Wei, P., Jiang, J., Cao, W., Bian, J., Chang, Y. (2020b) MESA: boost ensemble imbalanced learning "
     "with meta-sampler, Advances in Neural Information Processing Systems, 33.",
     "https://arxiv.org/abs/2010.08830"),
    ("Lopez-Rojas, E.A., Elmir, A., Axelsson, S. (2016) PaySim: a financial mobile money simulator for fraud "
     "detection, Proceedings of the 28th European Modeling and Simulation Symposium, Larnaca, pp.249-255.", None),
    ("Moro, S., Cortez, P., Rita, P. (2014) A data-driven approach to predict the success of bank telemarketing, "
     "Decision Support Systems, 62, pp.22-31.", "https://doi.org/10.1016/j.dss.2014.03.001"),
    ("Niculescu-Mizil, A., Caruana, R. (2005) Predicting good probabilities with supervised learning, Proceedings "
     "of the 22nd International Conference on Machine Learning, pp.625-632.",
     "https://doi.org/10.1145/1102351.1102430"),
    ("Park, S., Im, J. (2024) Experimental dataset for imbalanced classification: application of relabeling & "
     "raking algorithm (version 1), Mendeley Data.", "https://doi.org/10.17632/pb3jd7vz9z.1"),
    ("Park, S., Lee, H.-H., Im, J. (2024) Relabeling & raking algorithm for imbalanced classification, Expert "
     "Systems with Applications, 247, 123274.", "https://doi.org/10.1016/j.eswa.2024.123274"),
    ("Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., "
     "Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, "
     "M., Duchesnay, E. (2011) Scikit-learn: machine learning in Python, Journal of Machine Learning Research, "
     "12, pp.2825-2830.", "https://www.jmlr.org/papers/v12/pedregosa11a.html"),
    ("Redmond, M., Baveja, A. (2002) A data-driven software tool for enabling cooperative information sharing "
     "among police departments, European Journal of Operational Research, 141(3), pp.660-678.",
     "https://doi.org/10.1016/S0377-2217(01)00264-8"),
    ("Saito, T., Rehmsmeier, M. (2015) The precision-recall plot is more informative than the ROC plot when "
     "evaluating binary classifiers on imbalanced datasets, PLOS ONE, 10(3), e0118432.",
     "https://doi.org/10.1371/journal.pone.0118432"),
    ("Seiffert, C., Khoshgoftaar, T.M., Van Hulse, J., Napolitano, A. (2010) RUSBoost: a hybrid approach to "
     "alleviating class imbalance, IEEE Transactions on Systems, Man, and Cybernetics, Part A, 40(1), "
     "pp.185-197.", "https://doi.org/10.1109/TSMCA.2009.2029559"),
    ("Theephoowiang, K., Hanskunatai, A. (2025) A partition-based hybrid algorithm for effective imbalanced "
     "classification, Data, 10(4), 54.", "https://doi.org/10.3390/data10040054"),
    ("Wagner, D., Heider, D., Hattab, G. (2021) Mushroom data creation, curation, and simulation to support "
     "classification tasks, Scientific Reports, 11, 8134.", "https://doi.org/10.1038/s41598-021-87602-3"),
    ("Yeh, I.-C., Lien, C.-H. (2009) The comparisons of data mining techniques for the predictive accuracy of "
     "probability of default of credit card clients, Expert Systems with Applications, 36(2), pp.2473-2480.",
     "https://doi.org/10.1016/j.eswa.2007.12.020"),
]


def render(doc, blocks, lang, R):
    tables = {"summary": table_summary, "wilcoxon": table_wilcoxon, "holdout": table_holdout,
              "ablation": table_ablation, "ulb": table_ulb}
    for b in blocks:
        kind = b[0]
        if kind == "h":
            D.keep_with_next(D.para(doc, "Heading 2", b[1]))
        elif kind == "sub":
            D.keep_with_next(D.para(doc, "Subsection", b[1]))
        elif kind == "p":
            D.para(doc, "Text", b[1])
        elif kind == "eq":
            D.equation(doc, b[1], b[2])
        elif kind == "tcap":
            D.caption(doc, "Table Title", b[1])
        elif kind == "fcap":
            D.caption(doc, "Figure Caption", b[1])
        elif kind == "fig":
            D.figure(doc, b[1], b[2])
        elif kind == "note":
            D.note(doc, b[1], size=7.5)
        elif kind == "table":
            tables[b[1]](doc, lang, R)
        elif kind == "custom":
            b[1](doc, lang, R)


def build(lang, R):
    doc = D.new_document(BASE_DOCX)
    lab = F.LABEL[lang]
    D.para(doc, "Title", F.TITLE[lang])
    a = F.AUTHORS[lang]
    D.para(doc, "Authors", a[0])
    D.para(doc, "Authors", a[1], italic=True)
    D.para(doc, "Authors", a[2], italic=True)
    p = D.para(doc, "Normal", lab["abstract"], bold=True)
    p.paragraph_format.space_before = Pt(10)
    D.para(doc, "Abstract", abstract_text(lang, R))
    p = D.para(doc, "Normal", f"**{lab['keywords']}:** {F.KEYWORDS[lang]}", italic=True)
    p.paragraph_format.space_before = Pt(6)
    D.para(doc, "Normal", f"**{lab['jel']}:** {F.JEL}", italic=True)

    en = lang == "en"
    method = F.method(lang, D.m_frac)
    method = method[:2] + [("tcap", f"{lab['table']} 1. " + (
        "Design choices of Cross-Balance and the alternatives tested" if en else
        "Alegerile de proiectare ale Cross-Balance și alternativele testate")),
        ("custom", table_components)] + method[2:]
    design = F.design(lang)
    design = design[:2] + [("tcap", f"{lab['table']} 2. " + (
        "Datasets: rows (N), features after encoding (p) and minority share" if en else
        "Seturi de date: rânduri (N), variabile după codificare (p) și ponderea clasei minoritare")),
        ("custom", table_datasets),
        ("note", "* Simulated or synthetic data. Development sets above 50,000 rows are stratified subsamples; "
                 "ULB card fraud is used at full size. Sources: UCI repository (Kelly, Longjohn et al., n.d.) for "
                 "Communities & Crime, Credit default, Mammographic mass, Secondary mushroom and Bank marketing; "
                 "KEEL (Alcalá-Fdez, Fernández et al., 2011) for segment0 and shuttle-2 vs 5; Kaggle for Sparkov "
                 "card fraud (***, n.d.a; generator: ***, n.d.d), PaySim, Card fraud 2023 (***, n.d.b) and Fraud "
                 "transactions (***, n.d.c); Ailerons from the imbalanced-regression collection (***, n.d.e); "
                 "hold-out sets: Dal Pozzolo, Caelen et al. (2015) and Park and Im (2024)." if en else
                 "* Date simulate sau sintetice. Seturile de dezvoltare cu peste 50.000 de rânduri sunt "
                 "subeșantioane stratificate; ULB este folosit la dimensiune completă. Surse: depozitul UCI (Kelly, "
                 "Longjohn et al., n.d.) pentru Communities & Crime, Credit default, Mammographic mass, Secondary "
                 "mushroom și Bank marketing; KEEL (Alcalá-Fdez, Fernández et al., 2011) pentru segment0 și "
                 "shuttle-2 vs 5; Kaggle pentru Sparkov (***, n.d.a; generator: ***, n.d.d), PaySim, Card fraud 2023 "
                 "(***, n.d.b) și Fraud transactions (***, n.d.c); Ailerons din colecția pentru regresie "
                 "dezechilibrată (***, n.d.e); seturile de control: Dal Pozzolo, Caelen et al. (2015) și Park and "
                 "Im (2024).")] + design[2:]
    blocks = (F.intro(lang) + F.related(lang) + method + design + results_text(lang, R)
              + discussion_text(lang, R) + conclusion_text(lang, R) + appendix_text(lang, R))
    render(doc, blocks, lang, R)

    D.keep_with_next(D.para(doc, "Heading 2", "REFERENCES" if en else "BIBLIOGRAFIE"))
    for text, url in REFERENCES:
        D.reference(doc, text, url)
    out = HERE / f"CrossBalance_ECOFORUM_{lang.upper()}_rev1.docx"
    doc.save(out)
    return out


def main():
    R = collect()
    for lang in ("en", "ro"):
        print(build(lang, R))


if __name__ == "__main__":
    main()
