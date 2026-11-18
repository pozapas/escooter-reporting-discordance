"""The predictive benchmark.

The severity models in `severity.py` are explanatory. This is a separate question: how
well can severity be predicted at all from these data? The two are reported apart so that
neither borrows the other's credibility — a model can predict poorly and still estimate a
coefficient honestly, and a model can predict well while its coefficients mean nothing.

Three things this module is careful about, because each is a standard way a benchmark
overstates itself.

**Calibration is fitted inside training folds only.** The narrative cue probabilities are
calibrated; if that map were fitted on all records, every test fold would have influenced
the transformation applied to it, and the reported performance would be optimistic by an
amount nobody can quantify afterwards.

**Everything is reported across repeats, not from one split.** Ten repeats of stratified
five-fold cross-validation, with the mean and the standard deviation for every metric. On
77 records in the smallest class a single split is close to meaningless, and the spread
across repeats is the honest measure of what the number is worth.

**The comparison against the ordinal model uses the same folds.** Otherwise the benchmark
and the primary model are scored on different samples and the comparison says nothing.

SHAP values are reported as predictive salience on out-of-fold predictions. They are not
causal and are not described as importance.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import severity as sv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

REPEATS = 10
FOLDS = 5
SEED = 20260914

# A small, documented grid. It is small on purpose: with 522 records and 77 in the
# smallest class, a large grid searched by cross-validation would overfit the search
# itself, and the benchmark exists to bound predictability rather than to win a contest.
RF_GRID = ({"n_estimators": 500, "max_depth": None, "min_samples_leaf": 5},
           {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 10})
GB_GRID = ({"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3},
           {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 2})


def _metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                                 f1_score, precision_recall_fscore_support)

    pred = proba.argmax(axis=1)
    ka = len(sv.SEV_ORDER) - 1
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, pred, labels=list(range(len(sv.SEV_ORDER))), zero_division=0)

    is_ka = (y_true == ka).astype(int)
    brier = float(np.mean((proba[:, ka] - is_ka) ** 2))
    pr_auc = (float(average_precision_score(is_ka, proba[:, ka]))
              if 0 < is_ka.sum() < len(is_ka) else float("nan"))

    out = {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "pr_auc_KA": pr_auc,
        "brier_KA": brier,
    }
    for i, lvl in enumerate(sv.SEV_ORDER):
        out[f"precision_{lvl}"] = float(prec[i])
        out[f"recall_{lvl}"] = float(rec[i])
        out[f"f1_{lvl}"] = float(f1[i])
    return out


def _ordinal_proba(train: pd.DataFrame, test: pd.DataFrame,
                   columns: list[str]) -> np.ndarray | None:
    """Out-of-fold probabilities from the primary ordinal model, on the same folds."""
    try:
        fit = sv.fit_ordered_logit(train, columns, "bench")
        X, _ = sv._design(test, columns)
        return sv._probs(X, fit.beta, fit.cut)
    except Exception:                                             # noqa: BLE001
        return None


def run(frame: pd.DataFrame, columns: list[str], repeats: int = REPEATS,
        folds: int = FOLDS, seed: int = SEED) -> dict:
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.model_selection import RepeatedStratifiedKFold

    X = frame[columns].astype(float).to_numpy()
    y = pd.Categorical(frame["sev3"], categories=list(sv.SEV_ORDER),
                       ordered=True).codes
    if (y < 0).any() or np.isnan(X).any():
        raise SystemExit("benchmark requires a complete design matrix")

    models = {}
    for i, g in enumerate(RF_GRID):
        models[f"random_forest_{i + 1}"] = (RandomForestClassifier, g)
    for i, g in enumerate(GB_GRID):
        models[f"gradient_boosting_{i + 1}"] = (GradientBoostingClassifier, g)

    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats,
                                       random_state=seed)
    per_fold: dict[str, list[dict]] = {k: [] for k in models}
    per_fold["ordinal_logit"] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, (tr, te) in enumerate(splitter.split(X, y)):
            for name, (cls, grid) in models.items():
                m = cls(random_state=seed + i, **grid)
                m.fit(X[tr], y[tr])
                proba = m.predict_proba(X[te])
                if proba.shape[1] != len(sv.SEV_ORDER):
                    full = np.zeros((len(te), len(sv.SEV_ORDER)))
                    full[:, m.classes_] = proba
                    proba = full
                per_fold[name].append(_metrics(y[te], proba))

            p = _ordinal_proba(frame.iloc[tr], frame.iloc[te], columns)
            if p is not None:
                per_fold["ordinal_logit"].append(_metrics(y[te], p))

    summary = {}
    for name, rows in per_fold.items():
        if not rows:
            continue
        keys = rows[0].keys()
        summary[name] = {
            k: {"mean": round(float(np.nanmean([r[k] for r in rows])), 4),
                "sd": round(float(np.nanstd([r[k] for r in rows], ddof=1)), 4)}
            for k in keys
        }
        summary[name]["n_folds"] = len(rows)

    # The rate a model achieves by always predicting the largest class. Any benchmark
    # that does not clear this is reporting the class distribution, not a prediction.
    shares = np.bincount(y, minlength=len(sv.SEV_ORDER)) / len(y)
    summary["_baseline_majority_class"] = {
        "accuracy_if_always_predicting_BC": round(float(shares.max()), 4),
        "balanced_accuracy": round(1 / len(sv.SEV_ORDER), 4),
        "class_shares": {lvl: round(float(shares[i]), 4)
                         for i, lvl in enumerate(sv.SEV_ORDER)},
    }
    return {"repeats": repeats, "folds": folds, "seed": seed,
            "n": int(len(y)), "n_covariates": len(columns),
            "grid": {"random_forest": [dict(g) for g in RF_GRID],
                     "gradient_boosting": [dict(g) for g in GB_GRID]},
            "results": summary}


def shap_salience(frame: pd.DataFrame, columns: list[str], seed: int = SEED) -> dict:
    """Mean absolute SHAP values on out-of-fold predictions."""
    try:
        import shap
    except ImportError:
        return {"available": False,
                "note": "shap is not installed; predictive salience not computed"}

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    X = frame[columns].astype(float).to_numpy()
    y = pd.Categorical(frame["sev3"], categories=list(sv.SEV_ORDER),
                       ordered=True).codes
    total = np.zeros(len(columns))
    n_seen = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                       random_state=seed)
            m.fit(X[tr], y[tr])
            vals = shap.TreeExplainer(m).shap_values(X[te])
            arr = np.abs(np.asarray(vals))
            while arr.ndim > 2:
                arr = arr.mean(axis=0) if arr.shape[0] <= 5 else arr.mean(axis=-1)
            total += arr.mean(axis=0)
            n_seen += 1

    mean_abs = total / max(n_seen, 1)
    order = np.argsort(-mean_abs)
    return {
        "available": True,
        "computed_on": "out-of-fold predictions",
        "interpretation": ("predictive salience only; these are not causal effects and "
                           "are not described as importance"),
        "mean_abs_shap": {columns[i]: round(float(mean_abs[i]), 5) for i in order},
    }


if __name__ == "__main__":
    frame = pd.read_csv(DATA / "model_frame.csv")
    cols = sv.ladder(frame)["M1"]

    print("Predictive benchmark. PRELIMINARY: cues are the fused reference-run labels.")
    print(f"{REPEATS} repeats x {FOLDS} folds, {len(cols)} covariates, n = {len(frame)}")
    print()

    out = run(frame, cols)
    res = out["results"]
    base = res["_baseline_majority_class"]

    print(f"{'model':<22}{'macro F1':>12}{'bal. acc':>12}{'PR-AUC KA':>12}"
          f"{'Brier KA':>12}")
    for name, m in res.items():
        if name.startswith("_"):
            continue
        print(f"{name:<22}"
              f"{m['macro_f1']['mean']:>8.3f}±{m['macro_f1']['sd']:<4.3f}"
              f"{m['balanced_accuracy']['mean']:>8.3f}±{m['balanced_accuracy']['sd']:<4.3f}"
              f"{m['pr_auc_KA']['mean']:>8.3f}±{m['pr_auc_KA']['sd']:<4.3f}"
              f"{m['brier_KA']['mean']:>8.3f}±{m['brier_KA']['sd']:<4.3f}")
    print()
    print(f"always predicting the largest class gives accuracy "
          f"{base['accuracy_if_always_predicting_BC']} and balanced accuracy "
          f"{base['balanced_accuracy']}")
    print(f"class shares: {base['class_shares']}")

    out["shap"] = shap_salience(frame, cols)
    if out["shap"].get("available"):
        top = list(out["shap"]["mean_abs_shap"].items())[:8]
        print()
        print("highest predictive salience (mean |SHAP|, out of fold):")
        for k, v in top:
            print(f"  {k:<30} {v}")

    path = DATA / "benchmark.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print()
    print(f"written to {path}")
