"""Train, tune, evaluate and save PhishGuard.

    python -m phishguard.train            # full run, writes models/ and reports/
    python -m phishguard.train --quick    # skip cross-validation and ablation
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

from .data import build_dataset, grouped_split
from .features import FEATURE_NAMES, feature_matrix
from .model import MODEL_PATH, PhishGuard
from .urltools import normalize_url

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def recall_at_fpr(y, prob, max_fpr: float = 0.02) -> float:
    fpr, tpr, _ = roc_curve(y, prob)
    return float(tpr[fpr <= max_fpr].max())


def metrics_at(y, prob, threshold) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float((tp + tn) / len(y)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "false_positive_rate": round(float(fp / max(fp + tn, 1)), 4),
        "roc_auc": round(float(roc_auc_score(y, prob)), 4),
        "pr_auc": round(float(average_precision_score(y, prob)), 4),
        "recall_at_2pct_fpr": round(recall_at_fpr(y, prob, 0.02), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_for_fpr(y, prob, max_fpr: float) -> float:
    """Lowest threshold (so highest recall) whose false-positive rate is <= max_fpr."""
    fpr, _, thresholds = roc_curve(y, prob)
    ok = np.where(fpr <= max_fpr)[0]
    return float(min(thresholds[ok[-1]], 0.99))


def tune_threshold(train: pd.DataFrame, seed: int, max_fpr: float) -> float:
    """Pick the decision threshold on a domain-disjoint validation split.

    Objective: catch as much phishing as possible while flagging at most
    `max_fpr` of legitimate URLs. False alarms are what make users ignore a
    security tool, so they are the constraint, not an afterthought.
    """
    fit, val = grouped_split(train, test_size=0.2, seed=seed + 1)
    model = PhishGuard(seed=seed).fit(fit["url"], fit["label"])
    y, prob = val["label"].values, model.predict_proba(val["url"])
    t = threshold_for_fpr(y, prob, max_fpr)
    print(f"  validation: ROC-AUC={roc_auc_score(y, prob):.4f}, threshold={t:.3f} gives "
          f"recall={recall_score(y, prob >= t):.4f} at FPR<={max_fpr:.0%}")
    return t


def cross_validate(df: pd.DataFrame, seed: int, folds: int = 5) -> dict:
    """Domain-grouped k-fold CV: how stable is the score across different test sites?"""
    rng = np.random.default_rng(seed)
    domains = np.array(sorted(df["domain"].unique()))
    rng.shuffle(domains)
    fold_of = {d: i % folds for i, d in enumerate(domains)}
    fold = df["domain"].map(fold_of).values
    rows = []
    for k in range(folds):
        tr, te = df[fold != k], df[fold == k]
        model = PhishGuard(seed=seed).fit(tr["url"], tr["label"])
        rows.append(metrics_at(te["label"].values, model.predict_proba(te["url"]), 0.5))
        print(f"  fold {k + 1}/{folds}: ROC-AUC={rows[-1]['roc_auc']:.4f}  "
              f"recall@2%FPR={rows[-1]['recall_at_2pct_fpr']:.4f}")
    keys = ["accuracy", "precision", "recall", "f1", "false_positive_rate", "roc_auc",
            "pr_auc", "recall_at_2pct_fpr"]
    return {k: {"mean": round(float(np.mean([r[k] for r in rows])), 4),
                "std": round(float(np.std([r[k] for r in rows])), 4)} for k in keys}


def _char_baseline(train_urls, train_y, test_urls, test_y) -> dict:
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3,
                          max_features=300_000, sublinear_tf=True, dtype=np.float32)
    clf = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced")
    clf.fit(vec.fit_transform(train_urls), train_y)
    return metrics_at(np.asarray(test_y), clf.predict_proba(vec.transform(test_urls))[:, 1], 0.5)


def _lexical_baseline(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict:
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=48,
                                         l2_regularization=1.0, class_weight="balanced",
                                         random_state=seed)
    clf.fit(feature_matrix(train["url"]), train["label"])
    return metrics_at(test["label"].values,
                      clf.predict_proba(feature_matrix(test["url"]))[:, 1], 0.5)


def ablation(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
             model: PhishGuard, seed: int) -> list[dict]:
    """Show how much each design decision changes the honest score."""
    y = test["label"].values
    r_train, r_test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label"])
    p_final, p_text = model.component_proba(test["url"])
    rows = [
        ("Naive: raw URLs, random split (leaky)",
         _char_baseline(r_train["url"], r_train["label"], r_test["url"], r_test["label"])),
        ("Raw URLs, domain split (still leaky: scheme)",
         _char_baseline(train["url"], train["label"], test["url"], test["label"])),
        ("Char n-grams only, normalised, domain split",
         _char_baseline(train["url"].map(normalize_url), train["label"],
                        test["url"].map(normalize_url), test["label"])),
        ("Hand-crafted features only, domain split", _lexical_baseline(train, test, seed)),
        ("Text model (char + word) only, domain split", metrics_at(y, p_text, 0.5)),
        ("PhishGuard stacked model, domain split", metrics_at(y, p_final, model.threshold)),
    ]
    return [{"setup": name, **m} for name, m in rows]


def plots(y, model: PhishGuard, test: pd.DataFrame, seed: int) -> None:
    p_final, p_text = model.component_proba(test["url"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for name, p in [("PhishGuard (stacked)", p_final), ("Text model only", p_text)]:
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    axes[0].axvline(0.02, color="grey", ls=":", lw=1)
    axes[0].set(title="ROC curve (unseen domains), log x-axis", xlabel="False positive rate",
                ylabel="True positive rate", xscale="log", xlim=(1e-3, 1))
    axes[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    for ax in axes[:2]:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left")
        ax.grid(alpha=0.3)
    ConfusionMatrixDisplay(confusion_matrix(y, p_final >= model.threshold),
                           display_labels=["Legit", "Phishing"]).plot(ax=axes[2], colorbar=False)
    axes[2].set_title(f"Confusion matrix @ threshold {model.threshold:.2f}")
    fig.tight_layout()
    fig.savefig(REPORTS / "evaluation.png", dpi=130)
    plt.close(fig)

    sample = test.sample(min(3000, len(test)), random_state=seed)
    s_text = model.text.predict_proba(sample["url"].map(normalize_url).tolist())
    X = np.c_[feature_matrix(sample["url"]), s_text]
    names = FEATURE_NAMES + ["text_model_score"]
    imp = permutation_importance(model.meta, X, sample["label"], scoring="roc_auc",
                                 n_repeats=5, random_state=seed)
    order = np.argsort(imp.importances_mean)[-16:]
    fig, ax = plt.subplots(figsize=(8, 6.4))
    ax.barh([names[i] for i in order], imp.importances_mean[order],
            xerr=imp.importances_std[order], color="#3b6fb6")
    ax.set(title="What the meta model relies on (permutation importance)",
           xlabel="Mean drop in ROC-AUC when the feature is shuffled")
    fig.tight_layout()
    fig.savefig(REPORTS / "feature_importance.png", dpi=130)
    plt.close(fig)


def write_report(summary: dict) -> None:
    t = summary["test"]
    lines = [
        "# PhishGuard evaluation report", "",
        f"Trained {summary['trained_at']} on {summary['n_train']:,} URLs; tested on "
        f"{summary['n_test']:,} URLs from **{summary['n_test_domains']:,} domains never seen "
        "in training**. All scores are the pure ML model, without the reputation allowlist.",
        "", "## Held-out test set", "",
        "| Metric | Value |", "|---|---|",
        *[f"| {k.replace('_', ' ')} | {v} |" for k, v in t.items() if k != "confusion"],
        "", f"Confusion matrix: {t['confusion']}", "",
        f"Decision threshold {summary['threshold']}, chosen on validation domains for a "
        f"false-positive budget of {summary['max_fpr_target']:.0%}.", "",
    ]
    if summary.get("cross_validation"):
        cv = summary["cross_validation"]
        lines += ["## 5-fold domain-grouped cross-validation (threshold 0.5)", "",
                  "| Metric | Mean | Std |", "|---|---|---|",
                  *[f"| {k.replace('_', ' ')} | {v['mean']} | {v['std']} |" for k, v in cv.items()], ""]
    if summary.get("ablation"):
        lines += ["## Ablation: why the evaluation design matters", "",
                  "| Setup | Accuracy | F1 | ROC-AUC | Recall @ 2% FPR |", "|---|---|---|---|---|",
                  *[f"| {r['setup']} | {r['accuracy']} | {r['f1']} | {r['roc_auc']} | "
                    f"{r['recall_at_2pct_fpr']} |" for r in summary["ablation"]], "",
                  "The first two rows are how most tutorials evaluate. They are inflated: the "
                  "model memorises domains and the fact that benign URLs in these datasets "
                  "rarely carry a scheme or https.", ""]
    lines += ["## Recall by impersonated brand (PhishTank label)", "",
              "| Brand | Phishing URLs | Recall |", "|---|---|---|",
              *[f"| {b} | {r['n']} | {r['recall']} |" for b, r in summary["recall_by_brand"].items()],
              "", "![evaluation](evaluation.png)", "", "![features](feature_importance.png)", ""]
    (REPORTS / "report.md").write_text("\n".join(lines))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip cross-validation and ablation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-fpr", type=float, default=0.02,
                    help="false-positive budget used to pick the threshold (default 0.02)")
    args = ap.parse_args(argv)
    REPORTS.mkdir(exist_ok=True)
    start = time.time()

    print("Building dataset ...")
    df = build_dataset(seed=args.seed)
    train, test = grouped_split(df, test_size=0.2, seed=args.seed)
    print(f"  {len(df):,} URLs | train {len(train):,} ({train['domain'].nunique():,} domains) | "
          f"test {len(test):,} ({test['domain'].nunique():,} domains)")
    print(f"  class balance: {df['label'].mean():.1%} phishing")

    print("Tuning the decision threshold on validation domains ...")
    threshold = tune_threshold(train, args.seed, args.max_fpr)

    print("Training final model on the full training split ...")
    model = PhishGuard(seed=args.seed).fit(train["url"], train["label"])
    model.threshold = threshold
    model.load_reputation()

    y = test["label"].values
    prob = model.predict_proba(test["url"])
    test_metrics = metrics_at(y, prob, threshold)
    print("Test set (unseen domains):", json.dumps(test_metrics, indent=2))

    test = test.assign(probability=prob, predicted=(prob >= threshold).astype(int))
    errors = test[test["predicted"] != test["label"]].copy()
    errors["error"] = np.where(errors["label"] == 1, "missed_phishing", "false_alarm")
    errors.sort_values("probability", ascending=False)[
        ["error", "probability", "url", "source", "brand_target"]].to_csv(REPORTS / "errors.csv", index=False)

    phish = test[test["label"] == 1]
    top_brands = phish["brand_target"].value_counts().head(8).index
    recall_by_brand = {b: {"n": int((g := phish[phish["brand_target"] == b]).shape[0]),
                           "recall": round(float(g["predicted"].mean()), 4)} for b in top_brands}

    summary = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "n_train": len(train), "n_test": len(test),
        "n_test_domains": int(test["domain"].nunique()),
        "threshold": round(threshold, 4), "max_fpr_target": args.max_fpr,
        "test": test_metrics, "recall_by_brand": recall_by_brand,
    }
    if not args.quick:
        print("Running 5-fold domain-grouped cross-validation ...")
        summary["cross_validation"] = cross_validate(df, args.seed)
        print("Running ablation study ...")
        summary["ablation"] = ablation(df, train, test, model, args.seed)
        for r in summary["ablation"]:
            print(f"  {r['setup']:<46} acc={r['accuracy']:.4f} AUC={r['roc_auc']:.4f} "
                  f"recall@2%FPR={r['recall_at_2pct_fpr']:.4f}")

    print("Plotting ...")
    plots(y, model, test, args.seed)
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_report(summary)
    print(f"Saved model to {model.save(MODEL_PATH)}")
    print(f"Reports in {REPORTS}/  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
