#!/usr/bin/env python3
"""
Finding the Needle: a controlled comparison of machine-learning models and
data-balancing techniques for detecting rare fraudulent transactions.

Reproduces the experiment, tables and data figures of the paper:
  * 4 classifiers  x  5 balancing strategies  = 20 configurations
  * leakage-safe stratified 5-fold cross-validation (main protocol)
  * time-ordered 70/30 hold-out (robustness check)
  * metrics at the default threshold (0.5) and at a threshold tuned for F1
    on a validation slice of each training set

Usage
-----
    python run_experiment.py                      # run everything (about 30-40 min on 2 CPU cores)
    python run_experiment.py --analyse-only       # rebuild tables/figures from results/ only

Data
----
Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
and place it at data/creditcard.csv (or pass --data PATH).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEED = 42
N_FOLDS = 5
VAL_SIZE = 0.20          # validation slice of each training set, used only for threshold selection
TEMPORAL_TRAIN = 0.70    # earliest 70% of transactions train, latest 30% test
DEFAULT_THRESHOLD = 0.5

CLASSIFIERS = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
STRATEGIES = ["None", "Class weight", "RUS", "ROS", "SMOTE"]
TARGET = "Class"


# --------------------------------------------------------------------------- data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_raw, f_raw = len(df), int(df[TARGET].sum())
    df = df.drop_duplicates(keep="first").reset_index(drop=True)   # exact duplicate rows removed before any split
    n, f = len(df), int(df[TARGET].sum())
    print(f"Raw file:            {n_raw:,} transactions, {f_raw} frauds")
    print(f"After deduplication: {n:,} transactions, {f} frauds ({100 * f / n:.3f}%), "
          f"{n_raw - n:,} duplicate rows removed, imbalance {(n - f) / f:.0f} : 1")
    return df


# ----------------------------------------------------------------- models & balancing
def make_model(name: str, strategy: str, y_fit: np.ndarray):
    """Fixed hyperparameters (paper Table III). Class weighting is the only setting that varies."""
    weighted = strategy == "Class weight"
    cw = "balanced" if weighted else None
    if name == "Logistic Regression":
        # L2 penalty is the scikit-learn default; the scaler sits inside the pipeline so it is fitted on training data only.
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, class_weight=cw, random_state=SEED)),
        ])
    if name == "Decision Tree":
        return DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_leaf=5,
                                      class_weight=cw, random_state=SEED)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None, max_features="sqrt",
                                      class_weight=cw, n_jobs=-1, random_state=SEED)
    if name == "XGBoost":
        spw = float((y_fit == 0).sum() / (y_fit == 1).sum()) if weighted else 1.0
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", scale_pos_weight=spw,
                             eval_metric="logloss", n_jobs=-1, random_state=SEED)
    raise ValueError(name)


def rebalance(strategy: str, X: np.ndarray, y: np.ndarray):
    """Applied to the fit portion of a training set only - never to validation or test data."""
    if strategy in ("None", "Class weight"):
        return X, y
    if strategy == "RUS":
        return RandomUnderSampler(random_state=SEED).fit_resample(X, y)
    if strategy == "ROS":
        return RandomOverSampler(random_state=SEED).fit_resample(X, y)
    if strategy == "SMOTE":
        # Scale before neighbours are found, then transform back so every model sees the original units.
        scaler = StandardScaler().fit(X)
        Xs, ys = SMOTE(k_neighbors=5, random_state=SEED).fit_resample(scaler.transform(X), y)
        return scaler.inverse_transform(Xs), ys
    raise ValueError(strategy)


# ------------------------------------------------------------------------- metrics
def tune_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Threshold that maximises F1 on the validation slice."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    prec, rec = prec[:-1], rec[:-1]
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return float(thr[int(np.argmax(f1))])


def threshold_metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (tp and precision + recall) else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, accuracy=(tp + tn) / len(y),
                precision=precision, recall=recall, f1=f1, mcc=mcc)


def evaluate(name, strategy, X_tr, y_tr, X_te, y_te, protocol, fold):
    """Split the training set into fit (80%) and validation (20%) parts, rebalance the fit part,
    train, choose the threshold on validation, and score the untouched test set once."""
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=VAL_SIZE, stratify=y_tr,
                                                  random_state=SEED)
    X_bal, y_bal = rebalance(strategy, X_fit, y_fit)
    model = make_model(name, strategy, y_fit)
    t0 = time.perf_counter()
    model.fit(X_bal, y_bal)
    fit_seconds = time.perf_counter() - t0

    p_val = model.predict_proba(X_val)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    t_tuned = tune_threshold(y_val, p_val)

    row = dict(protocol=protocol, fold=fold, classifier=name, strategy=strategy,
               n_fit=len(y_bal), n_fit_fraud=int(y_bal.sum()), n_test=len(y_te), n_test_fraud=int(y_te.sum()),
               fit_seconds=round(fit_seconds, 2),
               pr_auc=average_precision_score(y_te, p_te), roc_auc=roc_auc_score(y_te, p_te),
               threshold_tuned=t_tuned)
    for tag, t in (("default", DEFAULT_THRESHOLD), ("tuned", t_tuned)):
        row.update({f"{k}_{tag}": v for k, v in threshold_metrics(y_te, p_te, t).items()})
    return row, p_te


# ---------------------------------------------------------------------- experiment
def run(df: pd.DataFrame, out: Path):
    X = df.drop(columns=[TARGET]).to_numpy(dtype=np.float64)
    y = df[TARGET].to_numpy(dtype=int)
    rows = []

    # Main protocol: stratified 5-fold CV, every transaction tested exactly once.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        for name in CLASSIFIERS:
            for strategy in STRATEGIES:
                row, _ = evaluate(name, strategy, X[tr], y[tr], X[te], y[te], "cv", fold)
                rows.append(row)
                print(f"[CV fold {fold}] {name:<20} {strategy:<13} PR-AUC {row['pr_auc']:.3f}  "
                      f"F1@0.5 {row['f1_default']:.3f}  F1 tuned {row['f1_tuned']:.3f}  ({row['fit_seconds']:.0f}s)")
        pd.DataFrame(rows).to_csv(out / "per_fold_results.csv", index=False)   # checkpoint after each fold

    # Robustness check: time-ordered hold-out (earliest 70% train, latest 30% test).
    order = np.argsort(df["Time"].to_numpy(), kind="stable")
    cut = int(TEMPORAL_TRAIN * len(order))
    tr, te = order[:cut], order[cut:]
    print(f"Temporal split: train {len(tr):,} ({int(y[tr].sum())} frauds), test {len(te):,} ({int(y[te].sum())} frauds)")
    preds = {"y_true": y[te]}
    for name in CLASSIFIERS:
        for strategy in STRATEGIES:
            row, p_te = evaluate(name, strategy, X[tr], y[tr], X[te], y[te], "temporal", 0)
            rows.append(row)
            preds[f"{name}|{strategy}"] = p_te
            print(f"[Temporal] {name:<20} {strategy:<13} PR-AUC {row['pr_auc']:.3f}  F1 tuned {row['f1_tuned']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(out / "per_fold_results.csv", index=False)
    pd.DataFrame(preds).to_csv(out / "temporal_test_predictions.csv.gz", index=False, compression="gzip")
    total = res["fit_seconds"].sum()
    print(f"\n{len(res)} model fits, {total / 60:.1f} minutes of training time in total")
    return res


# ------------------------------------------------------------------ tables (paper)
def build_tables(res: pd.DataFrame, df: pd.DataFrame, out: Path):
    cv = res[res.protocol == "cv"]
    tmp = res[res.protocol == "temporal"]
    n_legit, n_fraud = int((df[TARGET] == 0).sum()), int(df[TARGET].sum())
    md = []

    def add(fname, title, table, note=""):
        md.append(f"## {title}\n\n{table.to_markdown(index=False, floatfmt='.3f')}\n")
        if note:
            md.append(note + "\n")
        table.to_csv(out / f"{fname}.csv", index=False)

    # Table II - dataset after deduplication
    add("table2_dataset", "Table II: dataset after deduplication", pd.DataFrame({
        "Property": ["Transactions after removing duplicates", "Share of total", "Imbalance ratio (legitimate : fraud)"],
        "Legitimate": [f"{n_legit:,}", f"{100 * n_legit / len(df):.3f}%", f"{n_legit / n_fraud:.0f}"],
        "Fraud": [f"{n_fraud:,}", f"{100 * n_fraud / len(df):.3f}%", "1"],
        "Total": [f"{len(df):,}", "100%", ""]}))

    # Table IV - the accuracy paradox at the default threshold (counts pooled over the 5 test folds)
    pooled = cv.groupby(["classifier", "strategy"], sort=False)[
        ["tp_default", "fp_default", "fn_default", "tn_default"]].sum().reset_index()
    rows = [dict(Configuration="Baseline (flag nothing)", Accuracy=100 * n_legit / len(df), Recall=0.0,
                 Precision=np.nan, MCC=0.0, **{"Frauds caught": f"0 / {n_fraud}", "False alarms": 0})]
    for c, s in [("Logistic Regression", "None"), ("Logistic Regression", "SMOTE"), ("Decision Tree", "RUS"),
                 ("XGBoost", "RUS"), ("XGBoost", "None")]:
        r = pooled[(pooled.classifier == c) & (pooled.strategy == s)].iloc[0]
        tp, fp, fn, tn = r.tp_default, r.fp_default, r.fn_default, r.tn_default
        mcc = (tp * tn - fp * fn) / np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        rows.append(dict(Configuration=f"{c}, {s}", Accuracy=100 * (tp + tn) / (tp + fp + fn + tn),
                         Recall=tp / (tp + fn), Precision=tp / (tp + fp), MCC=mcc,
                         **{"Frauds caught": f"{tp} / {tp + fn}", "False alarms": int(fp)}))
    add("table4_accuracy_paradox", "Table IV: accuracy paradox at the default threshold (5-fold CV, pooled)", pd.DataFrame(rows),
        "Accuracy in %. Counts are summed over the five test folds; precision is undefined for the baseline.")

    # Fig. 2 data - PR-AUC and ROC-AUC, mean and SD over folds
    summ = cv.groupby(["classifier", "strategy"], sort=False).agg(
        pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
        roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std")).reset_index()
    add("fig2_data_auc_summary", "PR-AUC and ROC-AUC summary: mean and SD over 5 folds (data for Fig. 2)", summ)

    # Table V - Friedman tests (folds as blocks)
    def friedman(sub, metric, strategies):
        piv = sub.pivot(index="fold", columns="strategy", values=metric)[strategies]
        stat, p = friedmanchisquare(*[piv[s].to_numpy() for s in strategies])
        return stat, p
    rows = []
    for c in CLASSIFIERS:
        sub = cv[cv.classifier == c]
        no_rus = [s for s in STRATEGIES if s != "RUS"]
        s_all, p_all = friedman(sub, "pr_auc", STRATEGIES)
        s_nr, p_nr = friedman(sub, "pr_auc", no_rus)
        s_f1, p_f1 = friedman(sub, "f1_tuned", no_rus)
        rows.append({"Classifier": c, "All 5 strategies: chi2": s_all, "All 5 strategies: p": p_all,
                     "Without RUS: chi2": s_nr, "Without RUS: p": p_nr, "Without RUS, tuned F1: p": p_f1})
    add("table5_friedman", "Table V: Friedman tests comparing balancing strategies using PR-AUC and tuned F1", pd.DataFrame(rows),
        "Five folds per test, so power is low: a non-significant result means no detectable difference.")

    # Supporting statistics quoted in the text
    piv = cv[cv.strategy == "None"].pivot(index="fold", columns="classifier", values="pr_auc")[CLASSIFIERS]
    stat, p = friedmanchisquare(*[piv[c].to_numpy() for c in CLASSIFIERS])
    lines = [f"- Classifiers compared at strategy 'None' (PR-AUC): chi2 = {stat:.2f}, p = {p:.3f}",
             f"- XGBoost beat Random Forest on PR-AUC (strategy 'None') in {int((piv['XGBoost'] > piv['Random Forest']).sum())} of {N_FOLDS} folds"]
    for c in CLASSIFIERS:
        sub = cv[cv.classifier == c].pivot(index="fold", columns="strategy", values="pr_auc")
        last = int((sub.idxmin(axis=1) == "RUS").sum())
        beats = {s: int((sub[s] > sub["None"]).sum()) for s in ["Class weight", "ROS", "SMOTE"]}
        lines.append(f"- {c}: RUS ranked last on PR-AUC in {last}/{N_FOLDS} folds; folds beating 'None': {beats}")
    md.append("## Statistics quoted in the text\n\n" + "\n".join(lines) + "\n")

    # Table VI - threshold-dependent metrics, mean of 5 folds
    t6 = cv.groupby(["classifier", "strategy"], sort=False)[
        ["precision_default", "recall_default", "f1_default", "mcc_default",
         "precision_tuned", "recall_tuned", "f1_tuned", "mcc_tuned", "threshold_tuned"]].mean().reset_index()
    t6.columns = ["Classifier", "Strategy", "P @0.5", "R @0.5", "F1 @0.5", "MCC @0.5",
                  "P tuned", "R tuned", "F1 tuned", "MCC tuned", "Mean tuned threshold"]
    add("table6_threshold_metrics", "Table VI: threshold-dependent metrics at the default and tuned thresholds (mean of 5 folds)", t6)

    # Table VII - XGBoost frauds caught and false alarms, pooled over folds
    x = cv[cv.classifier == "XGBoost"].groupby("strategy", sort=False)[
        ["tp_default", "fp_default", "tp_tuned", "fp_tuned"]].sum().reindex(STRATEGIES)
    base_tp, base_fp = x.loc["None", "tp_default"], x.loc["None", "fp_default"]
    rows = []
    for s, r in x.iterrows():
        extra = r.tp_default - base_tp
        rows.append({"Strategy": s, "Caught @0.5": int(r.tp_default), "False alarms @0.5": int(r.fp_default),
                     "Extra frauds vs none": "-" if s == "None" else f"{extra:+d}",
                     "Extra false alarms per extra fraud": "-" if s == "None" or extra <= 0
                     else f"{(r.fp_default - base_fp) / extra:.1f}",
                     "Caught tuned": int(r.tp_tuned), "False alarms tuned": int(r.fp_tuned)})
    add("table7_xgboost_alerts", "Table VII: XGBoost frauds caught and false alarms (pooled over 5 test folds)", pd.DataFrame(rows),
        f"Out of {n_fraud} frauds and {n_legit:,} legitimate transactions.")

    # Table VIII - temporal hold-out
    t8 = pd.DataFrame({"Classifier": tmp.classifier, "Strategy": tmp.strategy, "PR-AUC": tmp.pr_auc,
                       "ROC-AUC": tmp.roc_auc, "F1 @0.5": tmp.f1_default, "F1 tuned": tmp.f1_tuned,
                       "Caught tuned": [f"{a} / {a + b}" for a, b in zip(tmp.tp_tuned, tmp.fn_tuned)],
                       "False alarms tuned": tmp.fp_tuned})
    n_te = int(tmp.n_test.iloc[0]); f_te = int(tmp.n_test_fraud.iloc[0])
    add("table8_temporal_holdout", "Table VIII: temporal hold-out results (train on first 70%, test on last 30%)", t8,
        f"Test period: {n_te:,} transactions, {f_te} frauds.")

    # Training-set sizes and run time quoted in Section IV
    sizes = cv.groupby("strategy", sort=False)[["n_fit", "n_fit_fraud"]].mean().reindex(STRATEGIES)
    md.append("## Training-set sizes (mean fit portion per CV fold) and run time\n\n"
              + sizes.round(0).astype(int).to_markdown() + "\n\n"
              + f"Total training time over {len(res)} fits: {res.fit_seconds.sum() / 60:.1f} minutes\n")

    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Tables written to {out / 'tables.md'}")
    return summ


# ------------------------------------------------------------------------- figures
# Categorical colours, one per strategy, assigned in a fixed order (colour-blind checked).
COLORS = dict(zip(STRATEGIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3de"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def build_figures(res: pd.DataFrame, summ: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelcolor": INK})

    # Fig. 2 - PR-AUC by classifier and strategy, mean +/- SD over folds
    fig, ax = plt.subplots(figsize=(7, 3.4))
    width = 0.16
    xs = np.arange(len(CLASSIFIERS))
    for k, s in enumerate(STRATEGIES):
        d = summ[summ.strategy == s].set_index("classifier").reindex(CLASSIFIERS)
        ax.bar(xs + (k - 2) * width, d.pr_auc_mean, width, yerr=d.pr_auc_sd, color=COLORS[s], label=s,
               edgecolor="white", linewidth=1.0, error_kw=dict(ecolor=INK, elinewidth=0.8, capsize=2))
    ax.set_xticks(xs, CLASSIFIERS)
    ax.set_ylabel("PR-AUC (mean ± SD, 5 folds)")
    ax.set_ylim(0, 1)
    _style(ax)
    ax.legend(ncol=5, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig2_pr_auc.{ext}", dpi=300)
    plt.close(fig)

    # Fig. 3 - logistic regression: precision/recall at 0.5 (rings) and tuned (diamonds)
    lr = res[(res.protocol == "cv") & (res.classifier == "Logistic Regression")].groupby("strategy", sort=False)[
        ["precision_default", "recall_default", "precision_tuned", "recall_tuned"]].mean().reindex(STRATEGIES)
    fig, ax = plt.subplots(figsize=(5, 4))
    for s, r in lr.iterrows():
        ax.annotate("", xy=(r.recall_tuned, r.precision_tuned), xytext=(r.recall_default, r.precision_default),
                    arrowprops=dict(arrowstyle="->", color=COLORS[s], lw=1.0, alpha=0.8))
        ax.scatter(r.recall_default, r.precision_default, s=60, facecolors="none", edgecolors=COLORS[s], linewidths=2)
        ax.scatter(r.recall_tuned, r.precision_tuned, s=60, marker="D", color=COLORS[s], edgecolors="white", linewidths=1)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    lo = min(0.5, float(lr[["recall_default", "recall_tuned"]].min().min()) - 0.05)
    ax.set_xlim(max(lo, 0.0), 1.0); ax.set_ylim(0, 1.0)
    _style(ax); ax.grid(axis="x", color=GRID, linewidth=0.6)
    handles = [Line2D([], [], marker="s", ls="", color=COLORS[s], label=s) for s in STRATEGIES]
    handles += [Line2D([], [], marker="o", ls="", mfc="none", mec=MUTED, label="Threshold 0.5"),
                Line2D([], [], marker="D", ls="", color=MUTED, label="Tuned threshold")]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig3_lr_thresholds.{ext}", dpi=300)
    plt.close(fig)

    # Fig. 4 - precision-recall curves on the temporal test set, 0.5 operating points marked
    pred_path = out / "temporal_test_predictions.csv.gz"
    if not pred_path.exists():
        print("temporal_test_predictions.csv.gz not found - skipping Fig. 4")
        return
    preds = pd.read_csv(pred_path)
    y = preds["y_true"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
    for ax, c in zip(axes.ravel(), CLASSIFIERS):
        for s in STRATEGIES:
            p = preds[f"{c}|{s}"].to_numpy()
            prec, rec, _ = precision_recall_curve(y, p)
            ax.plot(rec, prec, color=COLORS[s], lw=1.5, label=s)
            m = threshold_metrics(y, p, DEFAULT_THRESHOLD)
            if not np.isnan(m["precision"]):
                ax.scatter(m["recall"], m["precision"], s=36, color=COLORS[s], edgecolors="white", linewidths=1.5, zorder=3)
        ax.set_title(c, fontsize=9, color=INK)
        _style(ax); ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    for ax in axes[1]: ax.set_xlabel("Recall")
    for ax in axes[:, 0]: ax.set_ylabel("Precision")
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig4_temporal_pr_curves.{ext}", dpi=300)
    plt.close(fig)
    print(f"Figures written to {out}")


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/creditcard.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--analyse-only", action="store_true",
                    help="skip training; rebuild tables and figures from results/per_fold_results.csv")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    if args.analyse_only:
        res = pd.read_csv(args.out / "per_fold_results.csv", keep_default_na=False, na_values=[""])  # keep the strategy name "None" as text
    else:
        res = run(df, args.out)
    summ = build_tables(res, df, args.out)
    build_figures(res, summ, args.out)


if __name__ == "__main__":
    main()
