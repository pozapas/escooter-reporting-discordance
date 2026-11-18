"""Post hoc sensitivity: the four cues the 10-event rule kept out of every severity model.

The frozen screening (prespec section 3) admits a binary candidate only when each severity
class holds at least 10 riders with it. Five of the ten narrative-derived cues fail that
rule, so the cue-block test of M1 and every robustness fit built on it (ordinal family,
Firth, Bayesian, random parameters, label variants, M2) test only the five admitted cues.
This script fits the excluded cues with the same penalized and Bayesian estimators.

This analysis was specified after the primary results and after a look at the raw
cue-by-severity table, so it is reported as a post hoc sensitivity, not as a test of the
pre-specified hypothesis. Dooring (2 riders) is left out. Swerve or loss of control has
one KA rider, so its coefficient is reported but not interpreted.

What is written to `data/excluded_cues.json`:

* ``counts``         - riders with each excluded cue by severity class (screening.json);
* ``ordered_logit``  - M1 plus the four excluded cues (MX): LR test of the four against
  M1, each cue's LR test, Holm-adjusted p, coefficient, odds ratio and 95% interval, and
  the KA average marginal effect of distraction or impairment (1000 Krinsky-Robb draws);
* ``narrative_length`` - MX with standardized narrative length added, for distraction;
* ``firth`` and ``bayes`` - the Firth-type and normal(0, 1)-prior fits of
  `ordinal_alternatives.py` applied to MX;
* ``cv``             - repeated five-fold CV log loss of M0c, M1 and MX on the shared folds,
  with paired differences.

Run with:  python src/excluded_cues.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ordinal_alternatives as oa   # noqa: E402
import severity as sv               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "excluded_cues.json"

EXCLUDED = ("driveway_alley", "distraction_impairment", "swerve_loss_control", "wrong_way")
FOCUS = "distraction_impairment"


def _r(x, k=4):
    return None if x is None else round(float(x), k)


def counts() -> dict:
    scr = json.loads((DATA / "screening.json").read_text(encoding="utf-8"))
    rows = {r["candidate"]: r for r in scr["candidates"]}
    return {c: {"events_by_severity": rows[c]["events_by_severity"],
                "disposition": rows[c]["disposition"]}
            for c in EXCLUDED + ("dooring",)}


def coef_block(fit: sv.OrdinalFit, names) -> dict:
    se = np.sqrt(np.diag(fit.cov))[: len(fit.columns)]
    out = {}
    for c in names:
        i = fit.columns.index(c)
        b, s = fit.beta[i], se[i]
        out[c] = {"coef": _r(b, 3), "se": _r(s, 3),
                  "ci95": [_r(b - 1.96 * s, 3), _r(b + 1.96 * s, 3)],
                  "odds_ratio": _r(np.exp(b), 2),
                  "or_ci95": [_r(np.exp(b - 1.96 * s), 2), _r(np.exp(b + 1.96 * s), 2)],
                  "wald_p": _r(2 * stats.norm.sf(abs(b / s)), 4)}
    return out


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for k, (name, p) in enumerate(items):
        run = max(run, min(1.0, (m - k) * p))
        out[name] = _r(run, 4)
    return out


def cv_block(frame, specs: dict) -> dict:
    ll = {}
    for name, cols in specs.items():
        r = sv.cross_validated_log_loss(frame, cols)
        ll[name] = np.array([f["log_loss"] if f["ok"] else np.nan for f in r["fold_level"]])
    out = {"repeats": sv.CV_REPEATS, "folds": sv.CV_FOLDS, "seed": sv.CV_SEED,
           "mean_log_loss": {k: _r(np.nanmean(v), 4) for k, v in ll.items()},
           "paired": {}}
    for a, b in (("MX", "M0c"), ("MX", "M1"), ("M1", "M0c")):
        d = ll[a] - ll[b]
        out["paired"][f"{a}_minus_{b}"] = {
            "mean_difference": _r(np.nanmean(d), 4),
            "folds_first_better": int((d < 0).sum()), "folds": int(np.isfinite(d).sum())}
    return out


def main() -> None:
    frame = pd.read_csv(DATA / "model_frame.csv")
    L = sv.ladder(frame)
    m0c_cols, m1_cols = L["M0c"], L["M1"]
    mx_cols = m1_cols + list(EXCLUDED)

    m0c = sv.fit_ordered_logit(frame, m0c_cols, "M0c")
    m1 = sv.fit_ordered_logit(frame, m1_cols, "M1")
    mx = sv.fit_ordered_logit(frame, mx_cols, "MX")

    single = {}
    for c in EXCLUDED:
        f = sv.fit_ordered_logit(frame, m1_cols + [c], c)
        single[c] = sv.nested_lr_test(m1, f)
    holm_p = holm({c: single[c]["p"] for c in EXCLUDED})

    ames = sv.average_marginal_effects(mx, frame)
    focus_ame = ames[FOCUS]

    nl_cols = mx_cols + ["narrative_chars_z"]
    mx_nl = sv.fit_ordered_logit(frame, nl_cols, "MX+length")
    nl_drop = sv.fit_ordered_logit(frame, [c for c in nl_cols if c != FOCUS], "drop")

    # Fatal crashes are the ones most likely to carry toxicology and a full investigation,
    # so impairment may be written down more often when the outcome is worst. Refitting
    # without the K riders (KA becomes A only) tests whether they carry the association.
    nk = frame[frame["sev_kabco"] != "K"].reset_index(drop=True)
    mx_nk = sv.fit_ordered_logit(nk, mx_cols, "MX-noK")
    nk_drop = sv.fit_ordered_logit(nk, [c for c in mx_cols if c != FOCUS], "drop-noK")
    nk_ame = sv.average_marginal_effects(mx_nk, nk)[FOCUS]["KA"]
    k_rows = frame[frame["sev_kabco"] == "K"]

    # Firth-type and Bayesian fits of ordinal_alternatives, applied to MX.
    X, y = sv._design(frame, mx_cols)
    fi = oa.fit_firth(X, y, mx.raw.copy())
    se_f = np.sqrt(np.diag(fi["cov_nat"])[: X.shape[1]])
    j = mx_cols.index(FOCUS)
    b, _, diag, _ = oa.fit_bayes(X, y, np.ones(X.shape[1]), mx.cut, "excluded_cues")
    post = b[:, j]

    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": ("post hoc sensitivity, specified after the primary results and after "
                   "the raw cue-by-severity table was seen"),
        "inputs": ["data/model_frame.csv", "data/screening.json"],
        "excluded_cues_fitted": list(EXCLUDED),
        "dooring_left_out": "2 riders",
        "counts": counts(),
        "ordered_logit": {
            "model": "MX = M1 plus the four excluded cues, ordered logit",
            "n_covariates": len(mx_cols), "loglik": _r(mx.loglik, 3),
            "loglik_M1": _r(m1.loglik, 3), "loglik_M0c": _r(m0c.loglik, 3),
            "lr_four_vs_M1": sv.nested_lr_test(m1, mx),
            "lr_nine_cues_vs_M0c": sv.nested_lr_test(m0c, mx),
            "single_cue_lr_vs_M1": single, "single_cue_holm_p": holm_p,
            "coefficients": coef_block(mx, list(EXCLUDED)),
            "admitted_cues_in_MX": coef_block(mx, [c for c in m1_cols if c in sv.CUES]),
            "focus_ame": {"covariate": FOCUS, "KA": focus_ame.get("KA"),
                          "O": focus_ame.get("O"), "BC": focus_ame.get("BC")},
        },
        "narrative_length": {
            "control": "narrative_chars_z",
            "coefficient": coef_block(mx_nl, [FOCUS])[FOCUS],
            "lr_focus": sv.nested_lr_test(nl_drop, mx_nl),
        },
        "excluding_fatal": {
            "rule": "riders coded K removed; the KA class then holds A riders only",
            "n_riders": int(len(nk)), "n_fatal_removed": int(len(k_rows)),
            "fatal_with_focus_cue": int(k_rows[FOCUS].sum()),
            "focus_cells": {s: int(nk.loc[nk["sev3"] == s, FOCUS].sum()) for s in sv.SEV_ORDER},
            "coefficient": coef_block(mx_nk, [FOCUS])[FOCUS],
            "lr_focus": sv.nested_lr_test(nk_drop, mx_nk),
            "ame_KA": nk_ame,
        },
        "firth": {"coef": _r(fi["theta"][j], 3), "se": _r(se_f[j], 3),
                  "ci95": [_r(fi["theta"][j] - 1.96 * se_f[j], 3),
                           _r(fi["theta"][j] + 1.96 * se_f[j], 3)],
                  "odds_ratio": _r(np.exp(fi["theta"][j]), 2),
                  "or_ci95": [_r(np.exp(fi["theta"][j] - 1.96 * se_f[j]), 2),
                              _r(np.exp(fi["theta"][j] + 1.96 * se_f[j]), 2)],
                  "converged": fi["convergence"]["converged"]},
        "bayes": {"prior": "normal(0, 1) on every slope, raw coding",
                  "posterior_mean": _r(post.mean(), 3),
                  "ci95": [_r(np.percentile(post, 2.5), 3), _r(np.percentile(post, 97.5), 3)],
                  "odds_ratio": _r(np.exp(post.mean()), 2),
                  "or_ci95": [_r(np.exp(np.percentile(post, 2.5)), 2),
                              _r(np.exp(np.percentile(post, 97.5)), 2)],
                  "prob_positive": _r((post > 0).mean(), 3), "diagnostics": diag},
        "cv": cv_block(frame, {"M0c": m0c_cols, "M1": m1_cols, "MX": mx_cols}),
    }
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    o = out["ordered_logit"]
    print(f"wrote {OUT}")
    print("LR four vs M1:", o["lr_four_vs_M1"], " nine vs M0c:", o["lr_nine_cues_vs_M0c"])
    print("Holm:", holm_p)
    print("coefficients:", json.dumps(o["coefficients"], indent=1))
    print("focus AME:", o["focus_ame"])
    print("length control:", out["narrative_length"])
    print("excluding fatal:", out["excluding_fatal"])
    print("firth:", out["firth"], "\nbayes:", {k: v for k, v in out["bayes"].items() if k != "diagnostics"})
    print("cv:", out["cv"]["mean_log_loss"], out["cv"]["paired"])


if __name__ == "__main__":
    main()
